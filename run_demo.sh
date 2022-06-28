for i in {1..30}
do
    rm caption* image.png image2.png
    python flickr_probe_each_np.py  --image_idx $i
done
