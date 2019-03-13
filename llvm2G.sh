#!/bin/bash
DATA_DIR=data/
LLVM_DIR=${DATA_DIR}llvm/
PROJ_DIR=${DATA_DIR}proj/
mkdir -p ${LLVM_DIR}
for n in `seq 3 5` #10 25
do
    case $n in 
        `expr 3`)
            upper=`expr 300`
            ;;
        `expr 4`)
            upper=`expr 300`
            ;;
        `expr 5`)
            upper=`expr 300`
            ;;
    esac
	for cnt in `seq 1  ${upper}`
	do
	  funct=funct$n\_$cnt
	  ./graph-llvm-ir ${PROJ_DIR}${funct}.prj/solution_${funct}/.autopilot/db/a.o.3.ll --block \
	  -o ${LLVM_DIR}
  done
done
