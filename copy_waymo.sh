# while read p; do
#     cp $p /home/rdr2143/waymo-adv-dataset/single-failed-v1/yolo-labels/.
# done < waymo-single-failed-labels.txt

# while read p; do
#     cp $p /home/rdr2143/waymo-adv-dataset/single-failed-v1/.
# done < waymo-single-failed-images.txt

while read p; do
            rm $p
done < waymo-single-failed-remove.txt