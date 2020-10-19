from mmfashion.demo.test_retriever import get_retriever_model
import sys
sys.path.append("/mmfashion")
from mmfashion.mmfashion.utils import get_img_tensor


def get_model():
    model = get_retriever_model()

    # model = VGG16(weights='imagenet', include_top=False)
    # model = ResNet50(weights='imagenet', include_top=False)
    return model

model = get_model()
print(model)
# img_tensor = get_img_tensor('static/imgs/1163.jpg', False)
img_tensor = get_img_tensor('static/imgs (copy)/01_3_back.jpg', False)
print(img_tensor.shape)
query_feat = model(img_tensor, landmark=None, return_loss=False)
query_feat = query_feat.data.cpu().numpy()

print(query_feat.shape)
print(type(query_feat))