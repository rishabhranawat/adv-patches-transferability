while read p; do
    cp $p /home/rdr2143/inria-adv-dataset/single-failed-v2/yolo-labels/.
done < inria-single-failed-labels.txt

while read p; do
    cp $p /home/rdr2143/inria-adv-dataset/single-failed-v2/.
done < inria-single-failed-images.txt

while read p; do
	    rm $p
done < inria-single-failed-remove.txt
