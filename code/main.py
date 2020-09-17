import world
import utils
from data import loadAFL



#################################
# data and init
#################################
dataset = loadAFL(world.CONFIG['dataset'])
world.CONFIG['the number of nodes'] = dataset.num_nodes()
world.CONFIG['the number of classes'] = dataset.num_classes()
print(dataset)
print(utils.dict2table(world.CONFIG))
# --------------
from model import EmbeddingP
MODEL = EmbeddingP()


