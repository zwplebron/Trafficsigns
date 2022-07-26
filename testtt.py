#import IE/OpenCV/numpy/time model
from openvino.inference_engine import IECore, IENetwork
import cv2
import numpy as np
from time import time

#Configure inference computing device, IR file path, picture path
DEVICE = 'MYRIAD'
model_xml = 'saved_model.xml'
model_bin = 'saved_model.bin'
image_file = 'image3.jpg'
labels_map = ["people","speed 25","speed 40","red light","stop"]

#Initialize the plug-in and output the plug-in version number
ie = IECore()
ver = ie.get_versions(DEVICE)[DEVICE]
print("{descr}: {maj}.{min}.{num}".format(descr=ver.description, maj=ver.major, min=ver.minor, num=ver.build_number))
    
#Read IR model file
net = IENetwork(model=model_xml, weights=model_bin)

#Prepare input and output tensor
print("Preparing input blobs")
input_blob = next(iter(net.inputs))
out_blob = next(iter(net.outputs))
net.batch_size = 1

#Load the model into the AI inference computing device
print("Loading IR to the plugin...")
exec_net = ie.load_network(network=net, num_requests=1, device_name=DEVICE)

#Read in the picture
n, c, h, w = net.inputs[input_blob].shape
frame = cv2.imread(image_file)
initial_h, initial_w, channels = frame.shape
#Scale up the picture as required by the AI model
image = cv2.resize(frame, (w, h))  
#Change the image data structure from HWC to CHW according to the requirements of AI model
image = image.transpose((2, 0, 1))
print("Batch size is {}".format(n))

#Perform inference calculation
print("Starting inference in synchronous mode")
start = time()
res = exec_net.infer(inputs={input_blob: image})
end = time()
print("Infer Time:{}ms".format((end-start)*1000))

# Processing output
print("Processing output blob")
res = res[out_blob]

for obj in res[0][0]:
    # When the confidence index is greater than 0.5, the test results are displayed.
    if obj[2] > 0.5:
        xmin = int(obj[3] * initial_w)
        ymin = int(obj[4] * initial_h)
        xmax = int(obj[5] * initial_w)
        ymax = int(obj[6] * initial_h)
        class_id = int(obj[1])
        # Display confidence index, object label and bounding box
        color = (0,255,0) if class_id>1 else(255,0,0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        det_label = labels_map[class_id] if labels_map else str(class_id)
        cv2.putText(frame, det_label + ' ' + str(round(obj[2] * 100, 1)) + ' %', (xmin, ymin - 7),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)

print("Inference is completed")

#Display the processing result
cv2.imshow("Detection results",frame)
cv2.waitKey(0)
cv2.destroyAllWindows()