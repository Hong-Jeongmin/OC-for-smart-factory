import argparse
import torch
from codes.networks import EncoderHier
from codes import mvtecad
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from codes.utils import PatchDataset_NCHW, NHWC2NCHW, distribute_scores

import pickle
from pick_and_place_conv import pick_and_place

parser = argparse.ArgumentParser()
parser.add_argument('--obj', default='tile')
args = parser.parse_args()

__all__ = ['eval_encoder_NN_multiK', 'eval_embeddings_NN_multiK']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def infer(x, enc, K, S):
    dataset = PatchDataset_NCHW(x, K=K, S=S)
    loader = DataLoader(dataset, batch_size=len(x), shuffle=False, pin_memory=True)
    embs = np.empty((dataset.N, dataset.row_num, dataset.col_num, enc.D), dtype=np.float32)  # [-1, I, J, D]
    enc = enc.eval()
    with torch.no_grad():
        for xs, ns, iis, js in loader:
            xs = xs.to(device)
            embedding = enc(xs)
            embedding = embedding.detach().cpu().numpy()

            for embed, n, i, j in zip(embedding, ns, iis, js):
                embs[n, i, j] = np.squeeze(embed)
    return embs


def assess_anomaly_maps(obj, anomaly_maps):
    auroc_seg = mvtecad.segmentation_auroc(obj, anomaly_maps)

    anomaly_scores = anomaly_maps.max(axis=-1).max(axis=-1)
    auroc_det = mvtecad.detection_auroc(obj, anomaly_scores)
    return auroc_det, auroc_seg


#########################

def eval_encoder_NN_multiK(enc, obj):
    x_tr = mvtecad.get_x_standardized(obj, mode='train')
    x_te = mvtecad.get_x_standardized(obj, mode='test')

    embs64_tr = infer(x_tr, enc, K=64, S=16)
    embs64_te = infer(x_te, enc, K=64, S=16)

    x_tr = mvtecad.get_x_standardized(obj, mode='train')
    x_te = mvtecad.get_x_standardized(obj, mode='test')

    embs32_tr = infer(x_tr, enc.enc, K=32, S=4)
    embs32_te = infer(x_te, enc.enc, K=32, S=4)

    embs64 = embs64_tr, embs64_te
    embs32 = embs32_tr, embs32_te

    return eval_embeddings_NN_multiK(obj, embs64, embs32)


def eval_embeddings_NN_multiK(obj, embs64, embs32, NN=1):
    emb_tr, emb_te = embs64
    maps_64 = measure_emb_NN(emb_te, emb_tr, method='kdt', NN=NN)
    maps_64 = distribute_scores(maps_64, (256, 256), K=64, S=16)
    det_64, seg_64 = assess_anomaly_maps(obj, maps_64)

    emb_tr, emb_te = embs32
    maps_32 = measure_emb_NN(emb_te, emb_tr, method='ngt', NN=NN)
    maps_32 = distribute_scores(maps_32, (256, 256), K=32, S=4)
    det_32, seg_32 = assess_anomaly_maps(obj, maps_32)

    maps_sum = maps_64 + maps_32
    det_sum, seg_sum = assess_anomaly_maps(obj, maps_sum)

    maps_mult = maps_64 * maps_32
    det_mult, seg_mult = assess_anomaly_maps(obj, maps_mult)

    return {
        'det_64': det_64,
        'seg_64': seg_64,

        'det_32': det_32,
        'seg_32': seg_32,

        'det_sum': det_sum,
        'seg_sum': seg_sum,

        'det_mult': det_mult,
        'seg_mult': seg_mult,

        'maps_64': maps_64,
        'maps_32': maps_32,
        'maps_sum': maps_sum,
        'maps_mult': maps_mult,
    }


########################

def measure_emb_NN(emb_te, emb_tr, method='kdt', NN=1):
    from codes.nearest_neighbor import search_NN
    D = emb_tr.shape[-1]
    train_emb_all = emb_tr.reshape(-1, D)

    l2_maps, _ = search_NN(emb_te, train_emb_all, method=method, NN=NN)
    anomaly_maps = np.mean(l2_maps, axis=-1)

    return anomaly_maps



def do_evaluate_encoder_multiK(model_path, obj):
    from codes.inspection import eval_encoder_NN_multiK

    results = eval_encoder_NN_multiK(enc, obj)

    det_64 = results['det_64']
    seg_64 = results['seg_64']

    det_32 = results['det_32']
    seg_32 = results['seg_32']

    det_sum = results['det_sum']
    seg_sum = results['seg_sum']

    det_mult = results['det_mult']
    seg_mult = results['seg_mult']

    # print(
        # f'| K64 | Det: {det_64:.3f} Seg:{seg_64:.3f} | K32 | Det: {det_32:.3f} Seg:{seg_32:.3f} | sum | Det: {det_sum:.3f} Seg:{seg_sum:.3f} | mult | Det: {det_mult:.3f} Seg:{seg_mult:.3f} ({obj})')


#########################


def OC(obj, file, threshold, cropped):
    base_path = f'./ckpts/{obj}/'
    enc = EncoderHier(K=64, D=64).to(device)
    enc.load_state_dict(torch.load(base_path+'enchier.pkl', map_location=torch.device('cpu')))

    enc.eval()

    with open(base_path+'/emb_tr_64.pkl', 'rb') as f:
        embs64_tr = pickle.load(f)

    # with open(base_path+'emb_tr_32.pkl', 'rb') as f:
    #     embs32_tr= pickle.load(f)
    import PIL
    import time
    import matplotlib.pyplot as plt

    start = time.time()
    print(file)
    frame = PIL.Image.open(file+'.png')
    frame = frame.crop(cropped)
    frame = np.array(frame.resize((256, 256)))
    with open(base_path+'/mean.pkl', 'rb') as f:
        mean = pickle.load(f)
    data= (frame.astype(np.float32)-mean)/255 
    data = np.expand_dims(np.transpose(data, [2, 0, 1]), axis=0)
    embs64_te= infer(data, enc, 64, 16)
    maps_64 = measure_emb_NN(embs64_te, embs64_tr, method='kdt', NN=1)
    maps_64 = distribute_scores(maps_64, (256, 256), K=64, S=16)
    anomaly_scores = maps_64.max(axis=-1).max(axis=-1)
    print(anomaly_scores)
    # print(time.time()-start)
    

    #Visualization
    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(6,3)

    axes[0].imshow(frame)
    axes[0].set_axis_off()

    axes[1].imshow(maps_64.squeeze(), vmax=maps_64.max(), cmap='Reds')
    axes[1].set_axis_off()

    plt.tight_layout()
    plt.savefig(file+'_anomaly_maps.png')
    if anomaly_scores<threshold:
        print('Normal')
        print('Normal')
        print('Normal')
        print('Normal')
        print('Normal')
        print('Normal')
        return False
    else:
        print('Abnormal')
        print('Abnormal')
        print('Abnormal')
        print('Abnormal')
        print('Abnormal')
        return True
    


if __name__ == '__main__':
    import os
    import cv2
    import time

    # connection start
    # host_ip = '192.168.170.128'
    # host_port = 4840
    # clientSock = client_conn(host_ip, host_port)
    
    if not os.path.exists('result') :
        os.makedirs('result')
    
    cap = cv2.VideoCapture(2)
    
    width = int(cap.get(3)) # 가로 길이 가져오기 
    height = int(cap.get(4)) # 세로 길이 가져오기
    fps = 20
    
    fcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('output.avi', fcc, fps, (width, height), isColor=False) 
    print(out.isOpened())
    
    cnt = 1


    while (True) :
        ret, frame = cap.read()
        if ret :
            color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            out.write(color)
            cv2.imshow('frame', color)

            cv2.waitKey(500)
            cv2.imwrite('result/screenshot{}.png'.format(cnt), frame, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
            if cnt%3==0:
                defect = OC('tile', f'./result/screenshot{cnt}', 0.0000395, (150, 30, 550, 430))
                # client_send(clientSock, defect)
                # print("Result : ",defect)
                pick_and_place(defect)
            
            cnt = cnt + 1
    
            # if cv2.waitKey(1) & 0xFF == ord('q') : break
        else :
            print("Fail to read frame!")
            break

    # connection close
    # clientSock.close()

    cap.release()
    out.release()
    cv2.destroyAllWindows()
