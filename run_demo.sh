for i in {1..300}
do
    rm caption* image.png image2.png
    CUDA_VISIBLE_DEVICES=0 python flickr_nps.py  --image_idx $i
done
