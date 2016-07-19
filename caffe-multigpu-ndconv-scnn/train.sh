#!/usr/bin/env bash
./build/tools/caffe train -gpu 1 \
-solver action_recog_st_split1_xy_dummy_solver.prototxt 
