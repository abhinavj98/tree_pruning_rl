import torch
import torch.nn as nn
import cv2
output_conv = nn.Conv2d(3, 1, 3, padding = 1)
output_conv.bias.data.fill_(0.3)
# decoder = nn.Sequential(
#             nn.ConvTranspose2d(128, 128, 2, stride=2), # 128. 14, 14
#             nn.ReLU(),
#             nn.ConvTranspose2d(128, 64, 2, stride=2),  # b, 64, 28, 28
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, 32, 2, stride=2),  # b, 32, 56, 56
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 16, 2, stride=2),  # b, 16, 112, 112
#             nn.ReLU(),
#             nn.ConvTranspose2d(16, 8, 2, stride=2), # b, 8, 224, 224
#             nn.ReLU(),
#             nn.Conv2d(8, 3, 3, padding = 1), # b, 3, 224, 224
#             nn.ReLU(),
#             output_conv  # b, 1, 224, 224
#             #nn.ReLU()
#         )
output_conv = nn.ConvTranspose2d(16, 3, 3, stride=2, output_padding=1, padding=1)
output_conv.bias.data.fill_(0.3)
decoder = nn.Sequential(
    nn.ConvTranspose2d(128, 128, 3, stride=2, output_padding=1, padding = 1), # 128. 14, 14
    nn.ReLU(),
    nn.ConvTranspose2d(128, 64, 3, stride=2, output_padding=1, padding=1),  # b, 64, 28, 28
    nn.ReLU(),
    nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding=1),  # b, 32, 56, 56
    nn.ReLU(),
    nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1, padding=1),  # b, 16, 112, 112
    nn.ReLU(),
    output_conv,  # b. 1, 224, 224
)
# for i in decoder:
#     try:
#         i.weight.data.fill_(1)
#         i.bias.data.fill_(0)
#     except:
#         pass
input = torch.ones(128, 7, 7)
# input[0][0][0] = 1
input = torch.normal(0,1,(128, 7, 7))
output = decoder(input)

#print(min(output.view(-1)), max(output.view(-1)))

output = torch.transpose(output, 0, 2)
print(output.shape)
output = (output - min(output.view(224*224, 3,1)))/max(output.view(224*224,3,1))
print(output.shape)
cv2.imshow("output", output.detach().numpy()*255)
cv2.waitKey(0)