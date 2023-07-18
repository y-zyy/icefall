import os
import glob
import torch
import torchaudio
from tqdm import tqdm
from emb_demixing.speaker_extractor import TEACHER

file_save_path = '/home/djlee/database/enroll_15s_embs'

def emb_extractor():

    enroll_audios = glob.glob('/home/djlee/database/enroll_15s/**/*.wav')
    
    m = TEACHER()
    m = m.cuda()
    m = m.eval()
    
    for file in tqdm(enroll_audios):
        wav, _ = torchaudio.load(file)
        wav = wav.cuda()
        embs = m(wav)
        embs = embs.cpu().detach()
        
        spkr = file.split('/')[-2]
        file_path = file_save_path + '/' + spkr + '.pt'
        torch.save(embs, file_path)

if __name__ == '__main__':
    emb_extractor()
    
