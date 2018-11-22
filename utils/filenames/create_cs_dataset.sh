cities=`ls /media/mpoggi/Storage/ComputerVision/Dataset/Cityscapes/`
for c in $cities;
do
	for n in `ls /media/mpoggi/Storage/ComputerVision/Dataset/Cityscapes/$c/*_leftImg8bit* | cut -d'_' -f'2,3'`;
	do

		if [ -f /media/mpoggi/Storage/ComputerVision/Dataset/Cityscapes/$c/"$c"_"$n"_gtFine_labelIds.png ]; then
			echo /media/mpoggi/Storage/ComputerVision/Dataset/Cityscapes/$c/"$c"_"$n"_leftImg8bit.jpg /media/mpoggi/Storage/ComputerVision/Dataset/Cityscapes/$c/"$c"_"$n"_rightImg8bit.jpg /media/mpoggi/Storage/ComputerVision/Dataset/Cityscapes/$c/"$c"_"$n"_gtFine_labelIds.png >> $1
		else
			echo /media/mpoggi/Storage/ComputerVision/Dataset/Cityscapes/$c/"$c"_"$n"_leftImg8bit.jpg /media/mpoggi/Storage/ComputerVision/Dataset/Cityscapes/$c/"$c"_"$n"_rightImg8bit.jpg /media/mpoggi/Storage/ComputerVision/Dataset/Cityscapes/nil.nil	>> $1
		fi
	done
done	
