# skeletonize_vine
Pipeline to take in point clouds and return skeletal representation

Example use case:
```
python3 skeletonize.py --cloud-paths /path/to/vine-only/cloud.ply --save-dir /path/to/save/output/ --close-cycles
```

There are a variety of parameters that can be tweaked, with reasonable defaults for the grapevine project. In addition, there are some flags that do extra visualizations. To see the various options, run

```
python3 skeletonize.py --help
```
