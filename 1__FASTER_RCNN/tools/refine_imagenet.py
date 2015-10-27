import os
import xml.dom.minidom as minidom

class imagenet():
    def __init__(self, image_set, year):
        self._year = year
        self._image_set = image_set
        self._devkit_path = os.path.join( self._get_default_path(), 'devkit' )
        self._data_path = os.path.join( self._get_default_path(), 'Data', 'DET', self._image_set )
        self._annot_path = os.path.join( self._get_default_path(), 'Annotations', 'DET', self._image_set )
        self._image_ext = '.JPEG'
        self._image_index = self._load_image_set_index()
        assert os.path.exists(self._annot_path), \
                'IMAGENET annotation path does not exist: {}'.format(self._annot_path)
        assert os.path.exists(self._data_path), \
                'IMAGENET data path does not exist: {}'.format(self._data_path)
    
    def _get_default_path(self):
        """
        Return the default path where IMAGENET is expected to be installed.
        """
        return os.path.join('/data/IMAGENET', 'ILSVRC' + self._year)

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file. (val, test)
        """
        image_set_file = os.path.join(self._get_default_path(), 'ImageSets', 'DET', 
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.split()[0] for x in f.readlines()]
        return image_index
    
    def _get_data_from_tag(self, node, tag):
        return node.getElementsByTagName(tag)[0].childNodes[0].data

    def _load_imagenet_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the IMAGENET
        format.
        """
        filename = os.path.join(self._annot_path, index + '.xml')
        if os.path.exists(filename) is True : # if a annotation file exists
            # check annotation file
            with open(filename) as f:
                data = minidom.parseString(f.read())
            objs = data.getElementsByTagName('object')
            size = data.getElementsByTagName('size')
            if len(size) != 1 :
                print "FUCK ANNOT FILE SIZE"
                return False
            iw = float(self._get_data_from_tag(size[0], 'width')) 
            ih = float(self._get_data_from_tag(size[0], 'height')) 
            num_objs = len(objs)
            if num_objs is 0 :
                return False
            else :
                for ix, obj in enumerate(objs):
                    x1 = float(self._get_data_from_tag(obj, 'xmin')) 
                    y1 = float(self._get_data_from_tag(obj, 'ymin')) 
                    x2 = float(self._get_data_from_tag(obj, 'xmax')) 
                    y2 = float(self._get_data_from_tag(obj, 'ymax')) 
                    if x2 <= x1 or y2 <= y1 : # can't define bbox
                        return False
            return True
        else :
            return False


if __name__ == '__main__' :
    imnet = imagenet('train','2015')
    f = open('train.txt', 'w')
    cnt = 0
    for index in imnet._image_index :
        flag = imnet._load_imagenet_annotation(index)
        if flag is True :
            cnt += 1
            string = '{} {}\n'.format(index, cnt)
            f.write(string)
    f.close()
