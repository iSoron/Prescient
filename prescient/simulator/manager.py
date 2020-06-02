#  ___________________________________________________________________________
#
#  Prescient
#  Copyright 2020 National Technology & Engineering Solutions of Sandia, LLC
#  (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.
#  Government retains certain rights in this software.
#  This software is distributed under the Revised BSD License.
#  ___________________________________________________________________________

import abc
import weakref

### Manager Class
# abstract base class
# different types of managers
# every manager has a hook back into the simulator
class _Manager(abc.ABC):

    def __init__(self):
        self._simulator = None

    def set_simulator(self, simulator):
        self._simulator = weakref.ref(simulator)

    @property
    def simulator(self):
        return self._simulator

