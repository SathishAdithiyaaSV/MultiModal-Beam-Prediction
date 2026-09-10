"""Minimal PLY point-cloud reader/writer, used in place of Open3D.

Open3D publishes no wheels for the Python version available on this machine
(3.14), so the two calls needed to read DeepSense6G point clouds --
`read_point_cloud` / `write_point_cloud(..., write_ascii=True)` -- are provided
here on top of numpy.

Behaviour intentionally matches Open3D:
  * only the x/y/z vertex properties are kept; extra properties such as the
    `intensity` field present in the DeepSense6G clouds are dropped, exactly as
    Open3D's PointCloud does;
  * `write_ply` emits an ascii PLY with double-precision x/y/z.
"""
import numpy as np

class TruncatedPLYError(ValueError):
    """Raised when a PLY body holds fewer vertices than its header declares."""


_PLY_DTYPES = {
    "char": "i1", "uchar": "u1", "short": "i2", "ushort": "u2",
    "int": "i4", "uint": "u4", "float": "f4", "double": "f8",
    "int8": "i1", "uint8": "u1", "int16": "i2", "uint16": "u2",
    "int32": "i4", "uint32": "u4", "float32": "f4", "float64": "f8",
}


def read_ply(path):
    """Return the (N, 3) float64 xyz array of a PLY file.

    Supports ascii and binary_little_endian PLY with a single `vertex` element.
    """
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError(f"{path}: not a PLY file")
        fmt = None
        props = []          # (name, numpy dtype string)
        n_vertex = None
        in_vertex = False
        while True:
            line = fh.readline()
            if not line:
                raise ValueError(f"{path}: truncated header")
            tok = line.split()
            if not tok:
                continue
            key = tok[0]
            if key == b"format":
                fmt = tok[1].decode()
            elif key == b"element":
                in_vertex = tok[1] == b"vertex"
                if in_vertex:
                    n_vertex = int(tok[2])
            elif key == b"property" and in_vertex:
                props.append((tok[2].decode(), _PLY_DTYPES[tok[1].decode()]))
            elif key == b"end_header":
                break
        if n_vertex is None:
            raise ValueError(f"{path}: no vertex element")
        names = [p[0] for p in props]
        for axis in ("x", "y", "z"):
            if axis not in names:
                raise ValueError(f"{path}: vertex element has no '{axis}' property")

        if fmt == "ascii":
            if n_vertex == 0:
                return np.zeros((0, 3))
            flat = np.array(fh.read().split(), dtype=np.float64)
            want = n_vertex * len(props)
            if flat.size < want:
                raise TruncatedPLYError(
                    f"{path}: header declares {n_vertex} vertices "
                    f"({want} values) but body holds {flat.size} values")
            flat = flat[:want].reshape(n_vertex, len(props))
            idx = [names.index(a) for a in ("x", "y", "z")]
            return np.ascontiguousarray(flat[:, idx], dtype=np.float64)

        if fmt == "binary_little_endian":
            rec = np.dtype([(n, "<" + d) for n, d in props])
            buf = fh.read(n_vertex * rec.itemsize)
            if len(buf) < n_vertex * rec.itemsize:
                raise TruncatedPLYError(
                    f"{path}: header declares {n_vertex} vertices "
                    f"({n_vertex * rec.itemsize} bytes) but body holds {len(buf)} bytes")
            arr = np.frombuffer(buf, dtype=rec, count=n_vertex)
            return np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float64)

        raise ValueError(f"{path}: unsupported PLY format {fmt!r}")


def write_ply(path, xyz):
    """Write an (N, 3) array as an ascii PLY with double x/y/z."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        "end_header\n"
    )
    with open(path, "w") as fh:
        fh.write(header)
        np.savetxt(fh, xyz, fmt="%.12f")
