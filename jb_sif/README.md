# README

Build the docker image:
```
docker build -t laser .
```

Then use [MobyJenkins_](https://jenkins.apps.idmod.org/view/Container%20building/job/MobyJenkins_) to build and publish on COMPS. You can lookup the asset collection on the COMPS gui. When running on COMPS, make sure you update the command line to the name of the `.sif` image:
```
    command = CommandLine("singularity exec ./Assets/krosenfeld_idm_laser_c3b6f1c.sif python3 -m idmlaser.measles")
```