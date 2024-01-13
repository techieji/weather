run: image
	docker run --rm -it -v "C:\Users\STUDENT LOANER\Documents\weather:/root" -p 8888:8888 ml-nwp

image: data
	docker build --rm -f Dockerfile -t ml-nwp .

data:
	mkdir data
	cd data;\
	curl -Z --user anonymous: --remote-name 'ftp://ftp2.psl.noaa.gov/Datasets/ncep.reanalysis/pressure/{uwnd,vwnd,shum,air,hgt,../surface/pres.sfc}.[1999-2001].nc'

clean:
	rm data -rf
