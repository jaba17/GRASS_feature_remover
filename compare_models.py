
import cv2
from standalone.Pix2Pix import Pix2Pix
from standalone.VitGan import VitGan
from standalone.utils import tensor2im, determine_edges
import os
import argparse

config = {
    "pix2pix":  {"model_url": "weights/latest_net_G.pth"},
    "cycle_gan":  {"model_url": "weights/pix2pix.pth"},
    "vit_gan": {"model_url":  "weights/vit_gan.h5"}
}


input_path = "real_input/"
output_path = "real_output/"

if __name__ == "__main__":

    # Init argparser
    parser = argparse.ArgumentParser()
    parser.add_argument("--canny", help="Also calculate canny edge map", default=False)
    parser.add_argument("--save", help="Save image", default=False)
    args = parser.parse_args()

    # Define models
    pix2pix = Pix2Pix(config["cycle_gan"])
    cycle_gan = Pix2Pix(config["pix2pix"])
    vit_gan = VitGan(config["vit_gan"])

    for file_name in os.listdir(input_path):
        
        # Load and prepare image
        img_path = input_path+file_name
        input_image = pix2pix.load_image_as_tensor(img_path)
        input_image = tensor2im(input_image)
        
        # Inference
        pix2pix_output = pix2pix.infer(img_path)
        cycle_gan_output = cycle_gan.infer(img_path)
        vit_gan_output = vit_gan.infer(img_path)
        vit_gan_output = cv2.cvtColor(vit_gan_output[0], cv2.COLOR_BGR2RGB)


        # Display results        
        cv2.imshow("original", input_image)
        cv2.imshow("pix2pix", pix2pix_output)
        cv2.imshow("cyclegan", cycle_gan_output)
        cv2.imshow("vit-gan", vit_gan_output)

        if args.canny:
            original_canny = determine_edges(input_image)
            cv2.imshow("original_edges", original_canny)

            pix2pix_canny = determine_edges(pix2pix_output)
            cv2.imshow("pix2pix_edges", pix2pix_canny)
        cv2.waitKey()

        if args.save:
            cv2.imwrite(output_path+"/original/"+file_name, input_image)
            cv2.imwrite(output_path+"/pix2pix/"+file_name, pix2pix_output)
            cv2.imwrite(output_path+"/cyclegan/"+file_name, cycle_gan_output)
            cv2.imwrite(output_path+"/vit-gan/"+file_name, 255*vit_gan_output)
            
            if args.canny: 
                cv2.imwrite(output_path+"/original_canny/"+file_name, original_canny)
                cv2.imwrite(output_path+"/pix2pix_canny/"+file_name, pix2pix_canny)

