import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer_test import Visualizer
from pdb import set_trace as st
from util import html
import matplotlib.pyplot as plt

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
    for i, data in enumerate(dataset):
        #print(data['A'].shape)
        print(data['A_paths'])
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.display_current_results(visuals, 0)
        for image in range(opt.depthSize):
            im = visuals['real_A'][0][image]
            im_s = visuals['fake_B'][0][image]
            plt.imsave(os.path.join(web_dir, 'images', "inference_%s.png" % image), im, cmap='gray')
            plt.imsave(os.path.join(web_dir, 'images', "synthetic_%s.png" % image), im_s, cmap='gray')

    webpage.save()
