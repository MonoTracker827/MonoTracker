optimizer:
	cd src/optimizer; mkdir -p build; cd build; cmake ..; make -j8; cd ../../..

globalsfm:
	cd helper_func/global_sfm; mkdir -p build; cd build; cmake ..; make -j8; cd ../../..
