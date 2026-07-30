# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

# Allow skipping initialization for lightweight tools
if not os.environ.get('LIDRA_SKIP_INIT'):
    import sam3d_objects.init
