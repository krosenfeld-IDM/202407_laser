# README

## Local build
Build the docker image for debugging
```
docker build -t laser .
```

## COMPS

1. Update `ImageName` otherwise the push to COMPS will fail.

2. Use [MobyJenkins_](https://jenkins.apps.idmod.org/view/Container%20building/job/MobyJenkins_) to build and publish on COMPS (use idmod account). You can lookup the asset collection on the COMPS gui. 

3. When running on COMPS, make sure you update the command line to the name of the `.sif` image:
```
    command = CommandLine("singularity exec ./Assets/krosenfeld_idm_laser_c3b6f1c.sif python3 -m idmlaser.measles")
```