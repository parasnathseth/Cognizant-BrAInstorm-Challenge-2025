# pip install pillow numpy scikit-learn torch torchvision
import os, glob, json, argparse
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
import torch, torch.nn as nn
import torchvision.transforms as T
from torchvision import models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Featurizer(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # 512-d pooled features
    def forward(self, x):
        f = self.backbone(x).squeeze(-1).squeeze(-1)
        return nn.functional.normalize(f, dim=-1)

pre = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def embed(path, model):
    img = Image.open(path).convert("RGB")
    x = pre(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        v = model(x).cpu().numpy()[0]
    return v

def build_index(data_dir, out_prefix="soil_memory", neighbors=3):
    model = Featurizer().to(DEVICE).eval()
    labels, paths, vecs = [], [], []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    for c in classes:
        for fp in glob.glob(os.path.join(data_dir, c, "*")):
            try:
                v = embed(fp, model)
                labels.append(c); paths.append(fp); vecs.append(v)
            except Exception:
                pass

    X = normalize(np.stack(vecs))
    nbrs = NearestNeighbors(n_neighbors=min(neighbors, len(labels)), metric="cosine").fit(X)
    np.save(out_prefix + "_X.npy", X)
    with open(out_prefix + "_meta.json", "w") as f:
        json.dump({"labels": labels, "paths": paths, "classes": classes}, f)
    print(f"Built index: {len(labels)} images across {len(classes)} classes")

def predict(img_path, index_prefix="soil_memory", accept=0.78):
    X = np.load(index_prefix + "_X.npy")
    meta = json.load(open(index_prefix + "_meta.json"))
    labels = meta["labels"]
    nbrs = NearestNeighbors(n_neighbors=min(3, len(labels)), metric="cosine").fit(X)

    model = Featurizer().to(DEVICE).eval()
    q = embed(img_path, model).reshape(1,-1)
    dists, idxs = nbrs.kneighbors(q, return_distance=True)
    sims = 1 - dists[0]
    top_lbl, top_sim = labels[idxs[0][0]], float(sims[0])
    result = {"label": top_lbl if top_sim >= accept else "unknown", "similarity": top_sim,
              "neighbors": [{"label": labels[j], "similarity": float(s)} for j,s in zip(idxs[0], sims)]}
    print(result)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", help="Folder with class subfolders (data/…)", default=None)
    ap.add_argument("--predict", help="Path to image to classify", default=None)
    ap.add_argument("--accept", type=float, default=0.78)
    args = ap.parse_args()

    if args.build:
        build_index(args.build)
    if args.predict:
        predict(args.predict, accept=args.accept)
