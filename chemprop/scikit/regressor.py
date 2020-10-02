from chemprop.parsing import parse_train_args
from chemprop.train import cross_validate
from chemprop.utils import create_logger
from chemprop.parsing import parse_predict_args
from chemprop.train import make_predictions
from pathlib import Path
import os
import csv
import logging

class MorganRootedAtoms:
    FUNC = "morgan"
    def __init__(self, radius=3, nbits=256):
        self.radius = radius
        self.nbits = 256

    def command_line(self):
        return f"{self.FUNC}-{self.radius}-{self.nbits}"

class MorganCountRootedAtoms(MorganRootedAtoms):
    FUNC = "morgancounts"

class RDKitRootedAtoms:
    FUNC = "rdkit"
    def __init__(self, minPath=1, maxPath=7, fpSize=256):
        self.minPath = minPath
        self.maxPath = maxPath
        self.fpSize = fpSize

    def command_line(self):
        return f"{self.FUNC}-{self.minPath}-{self.maxPath}-{self.fpSize}"
        
class RDKitUnbranchedRootedAtoms(RDKitRootedAtoms):
    FUNC = "rdkitunbranched"

class AtomPairs:
    FUNC = "atompairs"
    def __init__(self, minLength=1, maxLength=30, nBits=256):
        self.minLength = minLength
        self.maxLength = maxLength
        self.nBits = nBits

    def command_line(self):
        return f"{self.FUNC}-{self.minLength}-{self.maxLength}-{self.nBits}"

def method_string_to_graph_invariant(method_string):
    """Returns a method string, like morgan-3-256 to the appropraite
    method class.

    This is generally used for command line interface validation.
    """
    if not method_string:
        return None
    
    groups = method_string.split("-")
    groups[0] = groups[0].lower()
    method = groups[0]
    if method == "morgan":
        try:
            radius, nbits = map(int, groups[1:])
        except ValueError:
            logging.error("morgan format is morgan-radius-nbits")
            raise
        
        func = MorganRootedAtoms(radius, nbits)

    elif method == "morgancounts":
        try:
            radius, nbits = map(int, groups[1:])
        except ValueError:
            logging.error("morgancounts format is morgancounts-radius-nbits")
            raise

        func = MorganCountRootedAtoms(radius, nbits)
        
    elif method == "rdkit":
        try:
            minPath, maxPath, fpSize = map(int, groups[1:])
        except ValueError:
            logging.error("rdkit format is rdkit-minPath-maxPath-fpSize")
            raise

        func = RDKitRootedAtoms(minPath, maxPath, fpSize)
        
    elif method == "rdkitunbranched":
        try:
            minPath, maxPath, fpSize = map(int, groups[1:])
        except:
            logging.error("rdkitunbranched format is rdkitunbranched-minPath-maxPath-fpSize")
            raise
            
        func = RDKitUnbranchedRootedAtoms(minPath, maxPath, fpSize)
        
    elif method == "atompairs":
        try:
            minLength, maxLength, nBits = map(int, groups[1:])
        except:
            logging.error("atompairs format is rdkitunbranched-minPath-maxPath-fpSize")
            raise
            
        func = AtomPairs(minLength, maxLength, nBits)

    else:
        raise ValueError("Could not parse method %s"%method)

    return func

PARSING_KEYWORDS = ['epochs', 'batch_size', 'warmup_epochs',
                    'init_lr', 'max_lr', 'final_lr', 'ensemble_size', 'hidden_size',
                    'bias', 'depth', 'dropout', 'activation', 'undirected',
                    'ffn_hidden_size',
                    'ffn_num_layers', 'atom_messages']


class ChempropRegressor:
    def __init__(self,
                 checkpoint_directory=None,
                 checkpoint_count=None,
                 rdkit_features=False,
                 graph_invariant_func=None,
                 quiet=False,
                 metric = "rmse", # ['auc', 'prc-auc', 'rmse', 'mae', 'mse',
                                  #  'r2', 'accuracy', 'cross_entropy'],
                 epochs=30,
                 batch_size=50,
                 warmup_epochs=2.0,
                 init_lr=1e-4,
                 max_lr=1e-3,
                 final_lr=1e-4,
                 ensemble_size=1,
                 hidden_size=300,
                 bias=False,
                 depth=3,
                 dropout=0.0,
                 activation="ReLU", #['ReLU', 'LeakyReLU', 'PReLU', 'tanh',
                                    #'SELU', 'ELU']
                 undirected=False,
                 ffn_hidden_size=None,
                 ffn_num_layers=2,
                 atom_messages=False
    ):
        self.checkpoint_directory = checkpoint_directory
        self.checkpoint_count = checkpoint_count        
        self.rdkit_features = rdkit_features
        self.graph_invariant_func = graph_invariant_func
        self.quiet = quiet

        self.quiet=quiet
        self.metric = metric
        
        self.epochs=epochs
        self.batch_size=batch_size
        self.warmup_epochs=warmup_epochs
        self.init_lr=init_lr
        self.max_lr=max_lr
        self.final_lr=final_lr
        self.ensemble_size=ensemble_size
        self.hidden_size=hidden_size
        self.bias=bias
        self.depth=depth
        self.dropout=dropout
        self.activation=activation
        
        self.undirected=undirected
        self.ffn_hidden_size=ffn_hidden_size
        self.ffn_num_layers=ffn_num_layers
        self.atom_messages=atom_messages       

    def _to_command_line(self):
        kws = vars(self)
        out = []
        for kw in PARSING_KEYWORDS:
            val = kws[kw]
            if type(val) == bool:
                if val:
                    out.append(f"--{kw}")
            elif val is None:
                pass
            else:
                out.append(f"--{kw}")
                out.append(str(val))
        return out
    
    def fit(self, smiles, scores):
        if len(smiles) != len(scores):
            raise ValueError(f"List of smiles ({len(smiles)}) is not equal to list of scores ({len(scores)})")
        
        checkpoint_directory = self.checkpoint_directory
        checkpoint_count = self.checkpoint_count
        
        if checkpoint_directory is not None and checkpoint_count is not None:
            directory = os.path.join(checkpoint_directory, f"checkpoint_{checkpoint_count:03d}")
            if os.path.exists(directory):
                logging.warning("Directory %s exists, overwriting", directory)
            self.directory = directory
        else:
            if checkpoint_directory is None:
                checkpoint_directory = os.path.abspath(os.curdir)
            if checkpoint_count is not None:
                directory = os.path.join(checkpoint_directory, f"checkpoint_{checkpoint_count:03d}")
                if os.path.exists(directory):
                    logging.warning("Directory %s exists, overwriting", directory)
            else:
                checkpoint_count = 0
                directory = os.path.join(checkpoint_directory,
                                         f"checkpoint_{checkpoint_count:03d}")
                while os.path.exists(directory):
                    checkpoint_count += 1
                    directory = os.path.join(checkpoint_directory,
                                             f"checkpoint_{checkpoint_count:03d}")

                os.mkdir(directory)
            
            # save the checkpoint stuff required for rerunning the model.
            self.directory = directory

        datapath = os.path.join(self.directory, "data.csv")
        with open(datapath, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['smiles', 'scores'])
            for row in zip(smiles, scores):
                writer.writerow(row)

        # we DON'T want chemprop to split a train test, that happens outside.
        command_args = ['--data_path', datapath,
                        '--separate_test_path', datapath,
                        '--save_dir', self.directory,
                        '--dataset_type', 'regression']
        if self.rdkit_features:
            command_args += ['--features_generator', 'rdkit_2d_normalized',  '--no_features_scaling']
        if self.graph_invariant_func:
            command_args += ['--graph_invariant_func',
                             self.graph_invariant_func.command_line()]

        #command_args += self._to_command_line()
        args = parse_train_args(command_args)
        logger = create_logger(name='train',
                               save_dir=args.save_dir, quiet=self.quiet)
        cross_validate(args, logger=logger)
        

    def predict(self, smiles):
        count = 0
        test_path = os.path.join(self.directory, f"preds-in-{count:03d}")
        preds_path = os.path.join(self.directory, f"preds-out-{count:03d}")
        while os.path.exists(test_path) and os.path.exists(preds_path):
            count += 1
            test_path = os.path.join(self.directory, f"prediction-f{count:03d}")
            preds_path = os.path.join(self.directory, f"preds-out-f{count:03d}")
        
        Path(test_path).touch()
        Path(preds_path).touch()
        
        with open(test_path, 'w') as f:
            f.write("smiles\n")
            for smi in smiles:
                assert smi
                f.write("%s\n"%smi)

        command_args = ['--test_path', test_path, '--preds_path', preds_path,
                        '--checkpoint_dir', self.directory]
        if self.rdkit_features:
            command_args += ['--features_generator', 'rdkit_2d_normalized',
                             '--no_features_scaling']

        args = parse_predict_args(command_args)
        make_predictions(args)

        scores = []
        with open(preds_path) as f:
            for i,groups in enumerate(csv.reader(f)):
                if i != 0:
                    smi,score = groups
                    scores.append(float(score))
        
        return scores
    
