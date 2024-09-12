import cv2
from PIL import Image
from standalone.Pix2Pix import Pix2Pix
from standalone.utils import determine_edges

pix2pix_config = {"model_url": "weights/latest_net_G.pth"}
video_url = "videos/video1.mp4"

if __name__ == "__main__":
    pix2pix = Pix2Pix(pix2pix_config)

    cap = cv2.VideoCapture(video_url)
    while cap.isOpened():
        
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        else: 

            frame_canny= determine_edges(frame)

            image = Image.fromarray(frame)
            output = pix2pix.infer_image(image)
            output_canny = determine_edges(output)

            cv2.imshow('input', frame)
            cv2.imshow('input_canny', frame_canny)
            cv2.imshow('output', output)
            cv2.imshow('output_canny', output_canny)
            if cv2.waitKey(1) == ord('q'):
                break
