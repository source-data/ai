# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import os
import pickle
import torch
from .models import HyperparametersCatStack, Container1d, CatStack1d, HyperparametersUnet, Container2d, Unet2d, Autoencoder1d
from copy import deepcopy, copy
from datetime import datetime
from zipfile import ZipFile
from typing import ClassVar
from shutil import rmtree


def export_model(model, path, filename):
    archive_path = None
    try:
        if torch.cuda.is_available():
            model = deepcopy(model.module)
            model.cpu()
        prefix = datetime.now().isoformat("-", timespec='minutes').replace(":", "-") 
        archive_path = os.path.join(path, prefix + '_' + filename + '.zip')
        model_path = os.path.join(path, prefix + '_' + filename + '_model.th')
        hp_path = os.path.join(path, prefix + '_' + filename + '_hp.pickle')
        torch.save(model.state_dict(), model_path)
        with open(hp_path, 'wb') as f:
            pickle.dump(model.hp, f)
        with ZipFile(archive_path, 'w') as myzip:
            myzip.write(model_path)
            os.remove(model_path)
            myzip.write(hp_path)
            os.remove(hp_path)
    except Exception as e: 
        print(f"MODEL NOT SAVED: {filename}, {model_path}")
        print(e)
    return archive_path

def unzip_model(model_path):
    with ZipFile(model_path) as myzip:
        myzip.extractall()
        for filename in myzip.namelist():
            _, ext = os.path.splitext(filename)
            if ext == '.th':
                model_path = filename
            elif ext == '.pickle':
                hp_path = filename
    return hp_path, model_path

def load_model_from_class(path, model_class: ClassVar):
    print(f"\n\nloading {path}\n\n")
    hp_path, model_path = unzip_model(path)
    with open(hp_path, 'rb') as hyperparams:
        hp = pickle.load(hyperparams)
    print(f"trying to build model ({model_class.__name__} with hyperparameters:")
    print(hp)
    model =  model_class(hp)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    os.remove(model_path)
    os.remove(hp_path)
    return model

def load_container(path, container: ClassVar, sub_module: ClassVar):
    print(f"\n\nloading {path}\n\n")
    hp_path, model_path = unzip_model(path)
    with open(hp_path, 'rb') as hyperparams:
        hp = pickle.load(hyperparams)
    print(f"trying to build model ({container.__name__} with {sub_module.__name__}) with hyperparameters:")
    print(hp)
    model =  container(hp, sub_module)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    os.remove(model_path)
    os.remove(hp_path)
    return model

def load_autoencoder(path):
    model = load_model_from_class(path, model_class = Autoencoder1d)
    return model

def self_test():
    try:
        os.mkdir('/tmp/test')
        x = torch.zeros(2, 1, 4)

        hpcs = HyperparametersCatStack(N_layers=2, kernel=7, padding=3, stride=1, in_channels=1, out_channels=1, hidden_channels=2, dropout_rate=0.1)
        c1dcs = Container1d(hpcs, CatStack1d)
        y = c1dcs(x)
        saved_path = export_model(c1dcs, '/tmp/test', 'test_1')
        print(f"saved {saved_path}")
        loaded = load_container(saved_path, Container1d, CatStack1d)
        print(loaded)

        hpun = HyperparametersUnet(nf_table=[2,2,2], kernel_table=[3,3], stride_table=[1,1,1], pool=True, in_channels=1, hidden_channels=2, out_channels=1, dropout_rate=0.1)
        c2dun = Container2d(hpun, Unet2d)
        y = c1dcs(x)
        saved_path = export_model(c2dun, '/tmp/test', 'test_2')
        print(f"saved {saved_path}")
        loaded = load_container(saved_path, Container2d, Unet2d)
        print(loaded)


        hpcs = HyperparametersCatStack(N_layers=2, kernel=7, padding=3, stride=1, in_channels=1, out_channels=1, hidden_channels=2, dropout_rate=0.1)
        a1dcs = Autoencoder1d(hpcs)
        y = a1dcs(x)
        saved_path = export_model(a1dcs, '/tmp/test', 'test_3')
        print(f"saved {saved_path}")
        loaded = load_autoencoder(saved_path)
        print(loaded)

    finally:
        rmtree('/tmp/test')
        print("Cleaned up.")

def main ():
   self_test()

if __name__ == '__main__':
    main()