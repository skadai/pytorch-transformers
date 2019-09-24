# 这个脚本用来根据subtype抽取评论并按照8:2切分训练/测试数据集

subtype=$1
task_dir=$2


SQUAD_DIR=/data/projects/$2

if [ ! -d $SQUAD_DIR/$subtype ];then
mkdir $SQUAD_DIR/$subtype
else
echo "文件夹已经存在"
fi

subtype_file=`echo $subtype|tr "_" " "| tr "." "/"`
echo $subtype_file

pos_num=$(cat $SQUAD_DIR/general.json|grep "${subtype_file}"|wc -l)
echo "正样本 $pos_num"

neg_num=$(echo "$pos_num*0.8" | bc)
neg_num=${neg_num%.*}
echo "负样本 $neg_num"


if [ -f $SQUAD_DIR/$subtype/$subtype.json ];then
rm $SQUAD_DIR/$subtype/$subtype.json
echo "老文件删除"

else
echo "文件不存在,无需删除"
fi
cat $SQUAD_DIR/general.json|grep "${subtype_file}" >> $SQUAD_DIR/$subtype/$subtype.json
# 添加负样本需要优先考虑同一个type下的其他subtype
# shuf $SQUAD_DIR/general.json|grep -v "${subtype_file}" |tail -n $neg_num >> $SQUAD_DIR/$subtype/$subtype.json
# shuf $SQUAD_DIR/$subtype/$subtype.json > $SQUAD_DIR/$subtype/shuf_$subtype.json

# total_num=`expr $pos_num + 0`
# echo $total_num

# train_num=$(echo "$total_num*0.8" | bc)
# train_num=${train_num%.*}
# dev_num=`expr $total_num - $train_num`

# echo "dev num is: ${dev_num}"
# echo "train num is: ${train_num}"

# 选择不再拆分了, 补充负样本之后再来拆分
# head -n ${train_num} $SQUAD_DIR/$subtype/shuf_$subtype.json > $SQUAD_DIR/$subtype/train.json
# tail -n ${dev_num} $SQUAD_DIR/$subtype/shuf_$subtype.json > $SQUAD_DIR/$subtype/dev.json



