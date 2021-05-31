import scipy
import logging
from .loadCora import load_citation
from .loadppi import load_saintdata
from .load_edge import load_edgedata

logger = logging.getLogger(__name__)
def load_adjacency_matrix(file):
    data = scipy.io.loadmat(file)
    logger.info("loading mat file %s", file)
    if 'Features' in data:
        data['Attributes']=data['Features']
        del data['Features']
    return data

class Datasets:
    def __init__(self):
        super(Datasets, self).__init__()

    def get_graph(self):
        graph = None
        return graph

    @classmethod
    def attributed(cls):
        raise NotImplementedError

class ACM(Datasets):
    def __init__(self):
        super(ACM, self).__init__()

    def get_graph(self):
        dir='data/ACM/ACM.mat'
        return load_adjacency_matrix(dir)

    @classmethod
    def attributed(cls):
        return True


class Flickr(Datasets):
    def __init__(self):
        super(Flickr, self).__init__()

    def get_graph(self):
        dir = 'data/Flickr/Flickr_SDM.mat'

        return load_adjacency_matrix(dir)

    @classmethod
    def attributed(cls):
        return True

class BlogCatalog(Datasets):
    def __init__(self):
        super(BlogCatalog, self).__init__()

    def get_graph(self):
        dir = 'data/BlogCatalog/BlogCatalog.mat'
        return load_adjacency_matrix(dir)

    @classmethod
    def attributed(cls):
        return True

class Cora(Datasets):
    def __init__(self):
        super(Cora, self).__init__()

    def get_graph(self):
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset_str="cora")
        data={"Network":adj,"Label":labels,"Attributes":features}
        return data

    @classmethod
    def attributed(cls):
        return True

class Citeseer(Datasets):
    def __init__(self):
        super(Citeseer, self).__init__()

    def get_graph(self):
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset_str="citeseer")
        data={"Network":adj,"Label":labels,"Attributes":features}
        return data

    @classmethod
    def attributed(cls):
        return True


class Citeseer(Datasets):
    def __init__(self):
        super(Citeseer, self).__init__()

    def get_graph(self):
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset_str="citeseer")
        data={"Network":adj,"Label":labels,"Attributes":features}
        return data

    @classmethod
    def attributed(cls):
        return True


class neil001(Datasets):
    def __init__(self):
        super(neil001, self).__init__()

    def get_graph(self):
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset_str="nell.0.001")
        data={"Network":adj,"Label":labels,"Attributes":features}
        return data

    @classmethod
    def attributed(cls):
        return True

class pubmed(Datasets):
    def __init__(self):
        super(pubmed, self).__init__()

    def get_graph(self):
        adj, features, labels, idx_train, idx_val, idx_test = load_citation(dataset_str="pubmed")
        data={"Network":adj,"Label":labels,"Attributes":features}
        return data

    @classmethod
    def attributed(cls):
        return True


class ppi(Datasets):
    def __init__(self):
        super(ppi, self).__init__()

    def get_graph(self):
        adj_full, adj_train, feats, new_label, role = load_saintdata('ppi')
        data={"Network":adj_full,"Label":new_label,"Attributes":feats}
        return data

    @classmethod
    def attributed(cls):
        return True

class reddit(Datasets):
    def __init__(self):
        super(reddit, self).__init__()

    def get_graph(self):
        adj_full, adj_train, feats, new_label, role = load_saintdata('reddit')
        data={"Network":adj_full,"Label":new_label,"Attributes":feats}
        return data

    @classmethod
    def attributed(cls):
        return True

class yelp(Datasets):
    def __init__(self):
        super(yelp, self).__init__()

    def get_graph(self):
        adj_full, adj_train, feats, new_label, role = load_saintdata('yelp')
        data={"Network":adj_full,"Label":new_label,"Attributes":feats}
        return data

    @classmethod
    def attributed(cls):
        return True


class ogbn_arxiv(Datasets):
    def __init__(self):
        super(ogbn_arxiv, self).__init__()

    def get_graph(self):
        adj_full, adj_train, feats, new_label, role = load_saintdata('ogbn-arxiv')
        data={"Network":adj_full,"Label":new_label,"Attributes":feats}
        return data

    @classmethod
    def attributed(cls):
        return True

class chameleon(Datasets):
    def __init__(self):
        super(chameleon, self).__init__()

    def get_graph(self):
        adj, feature,labels= load_edgedata('chameleon')
        data={"Network":adj,"Label":labels,"Attributes":feature}
        return data

    @classmethod
    def attributed(cls):
        return True

class cornell(Datasets):
    def __init__(self):
        super(cornell, self).__init__()

    def get_graph(self):
        adj, feature,labels= load_edgedata('cornell')
        data={"Network":adj,"Label":labels,"Attributes":feature}
        return data

    @classmethod
    def attributed(cls):
        return True

class texas(Datasets):
    def __init__(self):
        super(texas, self).__init__()

    def get_graph(self):
        adj, feature,labels= load_edgedata('texas')
        data={"Network":adj,"Label":labels,"Attributes":feature}
        return data

    @classmethod
    def attributed(cls):
        return True

class texas(Datasets):
    def __init__(self):
        super(texas, self).__init__()

    def get_graph(self):
        adj, feature,labels= load_edgedata('texas')
        data={"Network":adj,"Label":labels,"Attributes":feature}
        return data

    @classmethod
    def attributed(cls):
        return True

class squirrel(Datasets):
    def __init__(self):
        super(squirrel, self).__init__()

    def get_graph(self):
        adj, feature,labels= load_edgedata('squirrel')
        data={"Network":adj,"Label":labels,"Attributes":feature}
        return data

    @classmethod
    def attributed(cls):
        return True

class wisconsin(Datasets):
    def __init__(self):
        super(wisconsin, self).__init__()

    def get_graph(self):
        adj, feature,labels= load_edgedata('wisconsin')
        data={"Network":adj,"Label":labels,"Attributes":feature}
        return data

    @classmethod
    def attributed(cls):
        return True


class film(Datasets):
    def __init__(self):
        super(film, self).__init__()

    def get_graph(self):
        adj, feature, labels = load_edgedata('film')
        data = {"Network": adj, "Label": labels, "Attributes": feature}
        return data

    @classmethod
    def attributed(cls):
        return True