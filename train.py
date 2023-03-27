import logging

from src.handlers.trainer import Trainer
from src.handlers.evaluater import Evaluater

from src.utils.general import save_json, load_json
from src.utils.parser import get_model_parser, get_train_parser

# Load logger
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    model_parser = get_model_parser()
    train_parser = get_train_parser()

    # Parse system input arguments
    model_args, moargs = model_parser.parse_known_args()
    train_args, toargs = train_parser.parse_known_args()

    # Making sure no unkown arguments are given
    assert set(moargs).isdisjoint(toargs), f"{set(moargs) & set(toargs)}"

    logger.info(model_args.__dict__)
    logger.info(train_args.__dict__)
    
    trainer = Trainer(model_args.path, model_args)
    trainer.train(train_args)
    
    evaluater = Evaluater(model_args.path, device='cuda')
    preds  = evaluater.load_preds(train_args.dataset, 'test')
    labels = evaluater.load_labels(train_args.dataset, 'test')
    acc = evaluater.calc_acc(preds, labels)
    logger.info(f"test: {acc:.2f}")
