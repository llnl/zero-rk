#!/bin/bash

ret_status=0

sh run_serial.sh
if [ ! "$?" -eq 0 ]
then
  ret_status=1
fi

sh run_parallel.sh
if [ ! "$?" -eq 0 ]
then
  ret_status=1
fi

exit $ret_status

