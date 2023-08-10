import cv2
from google.colab.patches import cv2_imshow

#here add the necessary foreground and background to merge.
foreground = cv2.imread('/content/2_onto_foregroun_at_iteration_0.png')
background = cv2.imread('/content/4_onto_backgroun_at_iteration_0.png')
# Resize the foreground image to match the size of the background image
foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

# Define the weights for foreground and background images
alpha = 0.5  
beta = 0.5   

# Merge the foreground and background images
merged_image = cv2.addWeighted(foreground, alpha, background, beta, 0.0)

# Display the merged image
cv2_imshow( merged_image)
cv2.imwrite('merged15.jpg',merged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
