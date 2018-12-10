import unittest
import os

import configs
import tools

class TestConfigs(unittest.TestCase):

  def test_saveloadconfigs(self):
      config = configs.input_ProtoConfig()
      savepath = os.path.join(os.getcwd(), 'datasets', 'tmp')
      try:
          os.mkdir(savepath)
      except FileExistsError:
          pass
      print(config.__dict__)
      tools.save_config(config, savepath)
      tools.load_config(savepath)

  def test_configupdate(self):
      config = configs.input_ProtoConfig()
      savepath = os.path.join(os.getcwd(), 'datasets', 'tmp')
      try:
          os.mkdir(savepath)
      except FileExistsError:
          pass
      print(config.__dict__)
      tools.save_config(config, savepath)
      config2 = tools.load_config(savepath)
      config2.update(config)

if __name__ == '__main__':
    unittest.main()