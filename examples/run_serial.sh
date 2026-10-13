
examples_dir=$PWD
ret_status=0
for d in `grep -v '^#' serial_examples.txt`
do
  cd $d
  echo "Running test $d ..."
  sh run.sh >& run.log
  if [ ! "$?" -eq 0 ]
  then
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "   Example: $d failed.     "
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo ""
    ret_status=1
  fi
  cd $examples_dir
done

exit $ret_status

