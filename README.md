# Weather prediction model

## Setting up the development environment

### Docker

Install docker and make, then run the command `make run`. This should automatically download relevant data files into a new directory titled `data` and then create and run a docker image named `ml-nwp`. This will take a couple of minutes, but after the initial setup run, it should be a lot faster.

### Docker (old)

Install [docker](https://docs.docker.com/get-docker/). Run the command `./dev.sh`; this may take a minute to install and setup the environment, and you should be placed into a shell; subsequent entries will be substantially faster. Here, the directory `/root` (or `~` since the default user is root and I didn't change that) is linked to the directory where the script is run from; **all changes made in any other place will be deleted when the environment is exited**.

### Shell

You can try running the `setup.sh` script; this may or may not work. This is almost certainly outdated and will probably give you a strange environment.
