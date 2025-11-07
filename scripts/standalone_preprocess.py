import os
import glob
import ast
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.ndimage
from PIL import Image

class TomogramProcessor:
    def __init__(
        self,
        in_dir: str = "./data/raw/",
        out_dir: str = "./data/processed/",
        target_size=(128, 704, 704),  # (depth, height, width)
        kernel_size=7,  # must be odd
        kernel_sigma=2.0,
        label_downsample=8,
    ):
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        self.target_size = tuple(int(x) for x in target_size)
        self.kernel_size = int(kernel_size)
        self.kernel_sigma = float(kernel_sigma)
        self.label_downsample = int(label_downsample)

        assert self.kernel_size % 2 == 1, "kernel_size must be odd"

        # create output dirs
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "images").mkdir(exist_ok=True)
        (self.out_dir / "labels").mkdir(exist_ok=True)
        (self.out_dir / "meta").mkdir(exist_ok=True)

        self.kernel = self._make_kernel()

    def _make_kernel(self):
        k = self.kernel_size
        center = k // 2
        # z,y,x grid
        z, y, x = np.meshgrid(
            np.arange(k) - center, np.arange(k) - center, np.arange(k) - center, indexing="ij"
        )
        kernel = np.exp(-(z**2 + y**2 + x**2) / (2.0 * (self.kernel_sigma ** 2)))
        # normalize to max 1.0 (optional)
        kernel = kernel.astype(np.float32)
        kernel = kernel / kernel.max()
        return kernel

    def _load_volume(self, path: Path):
        # path can be a directory of slices or a .npy file
        if path.is_dir():
            # look for common image extensions, sort properly
            files = sorted(
                glob.glob(str(path / "*.png"))
                + glob.glob(str(path / "*.jpg"))
                + glob.glob(str(path / "*.jpeg"))
                + glob.glob(str(path / "*.tif"))
                + glob.glob(str(path / "*.tiff"))
            )
            if not files:
                raise FileNotFoundError(f"No image slices found in directory {path}")
            slices = [np.array(Image.open(f).convert("L")) for f in files]
            volume = np.stack(slices, axis=0)
        elif path.suffix.lower() == ".npy":
            volume = np.load(path)
        else:
            raise ValueError(f"Unsupported file type: {path}")
        return volume

    def _resize_volume(self, volume: np.ndarray):
        if volume.shape == self.target_size:
            return volume.astype(np.uint8)
        zoom = (
            self.target_size[0] / volume.shape[0],
            self.target_size[1] / volume.shape[1],
            self.target_size[2] / volume.shape[2],
        )
        resized = scipy.ndimage.zoom(volume, zoom, order=1)
        resized = np.clip(resized, 0, 255).astype(np.uint8)
        return resized

    def _coords_to_label(self, coords, label_shape):
        # coords: list of (z,y,x) in relative [0,1] coordinates
        label = np.zeros(label_shape, dtype=np.float32)
        kz = self.kernel.shape[0] // 2

        for c in coords:
            if len(c) != 3:
                continue
            # clamp inputs to [0,1]
            z_rel = min(max(float(c[0]), 0.0), 1.0)
            y_rel = min(max(float(c[1]), 0.0), 1.0)
            x_rel = min(max(float(c[2]), 0.0), 1.0)

            zc = int(round(z_rel * (label_shape[0] - 1)))
            yc = int(round(y_rel * (label_shape[1] - 1)))
            xc = int(round(x_rel * (label_shape[2] - 1)))

            # target ranges in label volume
            z0 = max(0, zc - kz)
            z1 = min(label_shape[0], zc + kz + 1)
            y0 = max(0, yc - kz)
            y1 = min(label_shape[1], yc + kz + 1)
            x0 = max(0, xc - kz)
            x1 = min(label_shape[2], xc + kz + 1)

            # corresponding kernel slices
            kz0 = z0 - (zc - kz)  # if z0 > zc-kz, shift start in kernel
            kz1 = kz0 + (z1 - z0)
            ky0 = y0 - (yc - kz)
            ky1 = ky0 + (y1 - y0)
            kx0 = x0 - (xc - kz)
            kx1 = kx0 + (x1 - x0)

            # add via maximum to preserve peaks
            label[z0:z1, y0:y1, x0:x1] = np.maximum(
                label[z0:z1, y0:y1, x0:x1],
                self.kernel[kz0:kz1, ky0:ky1, kx0:kx1],
            )
        return label

    def process_tomogram(self, path: str, coords=None):
        p = Path(path)
        volume = self._load_volume(p)
        orig_shape = volume.shape

        # resize to target
        proc_volume = self._resize_volume(volume)

        # prepare label if coords provided
        if coords is not None and len(coords) > 0:
            # downsampled label shape
            label_shape = (
                int(self.target_size[0] // self.label_downsample),
                int(self.target_size[1] // self.label_downsample),
                int(self.target_size[2] // self.label_downsample),
            )
            label = self._coords_to_label(coords, label_shape)
        else:
            label = None

        return {"volume": proc_volume, "label": label, "orig_shape": orig_shape}

    def save_processed(self, data, name):
        name = Path(name).stem
        img_path = self.out_dir / "images" / f"{name}.npy"
        np.save(str(img_path), data["volume"])

        if data.get("label") is not None:
            lab_path = self.out_dir / "labels" / f"{name}.npy"
            np.save(str(lab_path), data["label"])

        meta = {
            "orig_shape": data["orig_shape"],
            "proc_shape": data["volume"].shape,
            "has_label": data.get("label") is not None,
        }
        meta_path = self.out_dir / "meta" / f"{name}_meta.npz"
        np.savez(str(meta_path), **meta)

    def process_dataset(self, coords_csv: str = None):
        # Build mapping from basename -> coords list
        coords_map = {}
        if coords_csv:
            coords_df = pd.read_csv(coords_csv)
            # Expect a 'name' column and 'coordinates' column with Python-list strings
            if "name" not in coords_df.columns or "coordinates" not in coords_df.columns:
                raise ValueError("labels_new.csv must contain 'name' and 'coordinates' columns")
            for _, row in coords_df.iterrows():
                name = str(row["name"])
                raw = row["coordinates"]
                if pd.isna(raw):
                    coords_map[name] = None
                    continue
                try:
                    coords = ast.literal_eval(raw)
                    coords_map[name] = coords
                except Exception:
                    # try comma-separated flat numbers e.g. "z,y,x;z,y,x"
                    if isinstance(raw, str) and ";" in raw:
                        pts = []
                        for seg in raw.split(";"):
                            seg = seg.strip()
                            if not seg:
                                continue
                            nums = [float(x) for x in seg.replace("(", "").replace(")", "").split(",")]
                            if len(nums) == 3:
                                pts.append(tuple(nums))
                        coords_map[name] = pts
                    else:
                        coords_map[name] = None

        # gather input paths from in_dir
        candidates = []
        # add .npy files
        candidates += sorted(self.in_dir.glob("*.npy"))
        # add directories (tomo_* or any folder)
        for d in sorted(self.in_dir.iterdir()):
            if d.is_dir():
                candidates.append(d)

        results = []
        for p in tqdm(candidates, desc="Processing tomograms"):
            name = p.stem
            coords = coords_map.get(name, None)
            try:
                data = self.process_tomogram(p, coords)
                self.save_processed(data, name)
                results.append({"name": name, "success": True, "orig_shape": data["orig_shape"], "has_label": coords is not None})
            except Exception as e:
                results.append({"name": name, "success": False, "error": str(e)})
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Adjust paths and parameters here
    IN_DIR = "./data/raw"
    OUT_DIR = "./data/processed"
    LABELS_CSV = "./labels_new.csv"  # path to your labels CSV

    processor = TomogramProcessor(
        in_dir=IN_DIR,
        out_dir=OUT_DIR,
        target_size=(128, 704, 704),
        kernel_size=7,
        kernel_sigma=2.0,
        label_downsample=8,
    )

    summary = processor.process_dataset(coords_csv=LABELS_CSV)
    print("Summary:")
    print(summary)
