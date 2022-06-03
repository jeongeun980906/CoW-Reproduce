import torch
import CLIP as clip
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

start_layer = 11

def interpret(image, texts, model, device):
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.to(device) * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    '''
    only on final layer?
    '''
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
          continue
        grad = torch.autograd.grad(one_hot, [blk.attn_probs], retain_graph=True)[0].detach()
        cam = blk.attn_probs.detach()
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0).mean(dim=1)
        R = R + torch.bmm(cam, R)
    image_relevance = R[:, 0, 1:]
    return  image_relevance

def show_image_relevance(image_relevance, image, orig_image,visualize=True,store = False,name=None):
  # create heatmap from mask on image
  def show_cam_on_image(img, mask):
      heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
      heatmap = np.float32(heatmap) / 255
      cam = heatmap + np.float32(img)
      cam = cam /np.max(cam)
      return cam

  # plt.axis('off')
  # f, axarr = plt.subplots(1,2)
  # axarr[0].imshow(orig_image)

  image_relevance = image_relevance.reshape(1, 1, 7, 7)
  image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
  image_relevance = 15* image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
#   image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
  image = image[0].permute(1, 2, 0).data.cpu().numpy()
  image = (image - image.min()) / (image.max() - image.min())
  vis = show_cam_on_image(image, image_relevance)
  vis = np.uint8(255 * vis)
  vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
  # axar[1].imshow(vis)
  if visualize:
    plt.figure()
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(orig_image)
    axs[0].axis('off')
    axs[1].imshow(vis)
    axs[1].axis('off')
    if store:
        plt.savefig('./res/{}.png'.format(name))
    else:
        plt.show()

  # plt.imshow(vis)
  return image_relevance

clip.clip._MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}

class clip_grad_cam():
    def __init__(self,device='cuda'):
        self.model,self. preprocess = clip.load("ViT-B/32", device=device, jit=False)
        self.device = device
    
    def set_text(self,text):
        self.text = clip.tokenize([text]).to(self.device)
    
    def run(self,img,vis=True,store = False,name=None):
        shape_temp = img.shape[0]
        img_pil = Image.fromarray(img)
        img = self.preprocess(img_pil).unsqueeze(0).to(self.device)
        R_image = interpret(model=self.model, image=img, texts=self.text, device=self.device)
        image_relevance = show_image_relevance(R_image[0], img, orig_image=img_pil,visualize=vis,store=store,name=name)

        R_image = R_image.reshape(1,1,7,7)
        R_image = torch.nn.functional.interpolate(R_image, size=shape_temp, mode='bilinear')
        # print(R_image.shape)
        return R_image[0][0].cpu().numpy(),image_relevance