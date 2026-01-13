# Copyright(C) 2026 Advanced Micro Devices, Inc. All rights reserved. Portions of this file consist of AI-generated content.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from core.tools.inference_pipe import init_pipe
from torchvision.utils import save_image
import os

def test():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    
    # Test GAN
    print("Testing GAN Model...")
    try:
        pipe = init_pipe(device, dtype, model_type='gan')
        prompt = 'cute toy owl made of suede, geometric accurate, relief on skin, plastic relief surface of body, intricate details, cinematic'
        output = pipe(prompt=prompt) # defaults to 3 steps
        
        # Save result
        output.images[0].save("out_pipe_gan.png")
        print("Saved out_pipe_gan.png")
    except Exception as e:
        print(f"GAN model failed (likely missing checkpoint): {e}")

    # Test Joint GAN
    print("\nTesting Joint GAN Model...")
    try:
        pipe_joint = init_pipe(device, dtype, model_type='joint_gan')
        prompt = 'cute toy owl made of suede, geometric accurate, relief on skin, plastic relief surface of body, intricate details, cinematic'
        output_joint = pipe_joint(prompt=prompt) # defaults to 6 steps
        output_joint.images[0].save("out_pipe_joint_gan.png")
        print("Saved out_pipe_joint_gan.png")
    except Exception as e:
        print(f"Joint GAN model failed (likely missing checkpoint): {e}")

if __name__ == "__main__":
    test()
