if [ $# != 2 ] ; then
	echo "USAGE: $0 dic_path Entity_type"
	echo " e.g.: $0 "body中文身体部位名称.dic  "BODY"
	exit 1;
fi
cat $1  |awk -v  type="$2" '{print($0,type )}'
