# -*- coding: utf-8 -*-
#T. Lemberger, 2018

import os
import pickle
import torch
from torch import nn
from .models import Hyperparameters, HyperparametersCatStack, Container, Container1d, CatStack1d, HyperparametersUnet, Container2d, Unet2d, Autoencoder1d
from copy import deepcopy, copy
from datetime import datetime
from zipfile import ZipFile
from typing import ClassVar, Tuple
from shutil import rmtree


def export_model(model: nn.Module, path: str, filename: str) -> str:
    """
    A function to save a model and its hyperparameters to disk in a single archive.
    All the files are prefixed with the datetime.
    The model state is saved to a <prefix>_<filename>_model.th and the Hyperparameter object is pickeled to <prefix>_<filename>_hp.pickle
    Both files a zipped together into <prefix>_<filename>.zip

    Params:
        model (nn.Module): the trained model
        path (str): the destination path
        filename (str): the basedname for the archive

    Returns:
        the full path to the archive
    """
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

def unzip_model(model_path: str) -> Tuple[str, str]:
    """
    A small utility to unzip a saved model archive.

    Params:
        model_path (str): path to the archive

    Returns:
        hp_path, model_path (str, str): the path to the pickeld Hyperparameter object and the save model state.
    """

    with ZipFile(model_path) as myzip:
        myzip.extractall()
        for filename in myzip.namelist():
            _, ext = os.path.splitext(filename)
            if ext == '.th':
                model_path = filename
            elif ext == '.pickle':
                hp_path = filename
    return hp_path, model_path


def load_model_from_class(path:str, model_class: ClassVar) -> nn.Module:
    """
    Generic function to load a model of type model_class.

    Params:
        path (str): the path to the model zip archive

    Returns:
        model (nn.Model): the trained model of type model_class
    """

    print(f"\n\nloading {path}\n\n")
    hp_path, model_path = unzip_model(path)
    with open(hp_path, 'rb') as hyperparams:
        hp = pickle.load(hyperparams)
    print(f"trying to build model {model_class.__name__} with hyperparameters:")
    print(hp)
    model =  model_class(hp)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    os.remove(model_path)
    os.remove(hp_path)
    return model


def load_container(path: str, container: ClassVar, sub_module: ClassVar) -> Container:
    """
    A function to load a Container model.

    Params:
        path (str): path to the model zip archive.
        container (ClassVar): the subclass of Container (Container1d or Container2d)
        sub_module (ClassVar): the submodule class (CatStack or Unet, 1d or 2d version) to be plugged into the Container.

    Returns:
        the traind model (Container)
    """

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
    """
    A function to load an Autoencoder1d model.

    Params:
        path (str): path to the model zip archive.

    Returns:
        the traind model (Autoencode1d)
    """
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