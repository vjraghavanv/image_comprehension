import boto3
from skimage import io

##image analyzer function
def imageAnalyzer(input_img):
    
    rek_client = boto3.client('rekognition')

    print(input_img)
    with open(input_img, 'rb') as image:
        response = rek_client.detect_labels(Image={'Bytes': image.read()})

    labels = response['Labels']
    print(f'Found {len(labels)} labels in the image:')
    label_names = ''
    for label in labels:
        name = label['Name']
        confidence = label['Confidence']
        #print(f'> Label "{name}" with confidence {confidence:.2f}')
        if confidence>95:
            print(name + "|" + str(confidence))
            label_names = label_names + name + ","

    return label_names
 
input_img = 'input_img.jpg'
img = io.imread(input_img)
io.imshow(img)

label_names = imageAnalyzer(input_img)
print(label_names)