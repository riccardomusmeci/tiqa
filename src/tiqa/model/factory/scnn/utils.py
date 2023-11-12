from typing import Union, cast

import torch.nn as nn


def weight_init(net: Union[nn.Sequential, nn.Linear]) -> None:
    """Function for the weights initialization according Kaiming method.

    Args:
        net (Union[nn.Sequential, nn.Linear]): _description_

    Raises:
        ValueError: if no module is found
        ValueError: if no bias is found
    """

    for m in net.modules():
        if m is None:
            raise ValueError()
        elif isinstance(m, nn.Conv2d):
            m = cast(nn.Conv2d, m)
            nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
            if m.bias is not None:
                m.bias.data.zero_()
            else:
                raise ValueError()
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight.data, nonlinearity="relu")
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
