ATTEMPTS

#1
- discriminator only on full generation
- use this reward for all generation steps
- generator didn't learn

#2
- discriminator only on full generation
- use this reward for final generation step; zero for others
- generator didn't learn

#3
- discriminator on each generation step
- seems to be working better

#4
- fix reinforce logic
- i think the training logic is correct now
- on to hyperparameter search

#5
- training
    - 100 epochs to 1000 epochs
    - discriminator loss from cross-entropy to MSE
    - add onecycle LR scheduler
        - momentum cycle from 0.4 to 0.9
    - gamma from 0.8 to 0.5
    - generator LR from 6e-5 to 1e-4
    - discriminator LR from 3e-3 to 1e-2
- models
    - hidden size from 64 to 16
    - intermediate size from 256 to 64
    - attention heads from 8 to 4
    - rope theta from 10000 to 500
    - tie word embeddings in generator
    - discriminator layers from 2 to 1
    - discriminator attention dropout from 0.0 to 0.5
- generator starts learning for a bit now then delearns as the discriminator gets better
- hopefully the generator keeps learning once the discriminator plateaus
- batch size is 64 and only using ~3GB of VRAM
    - can increase batch size and/or make generator bigger

#6
- batch size from 64 to 128
- generator
    - hidden size from 16 to 32
    - intermediate size from 64 to 128
    - attention heads from 4 to 8

#7
- context length from 128 to 32
- batch size from 128 to 256
- generator
    - hidden size from 32 to 128
    - intermediate size from 128 to 512
    - layers from 2 to 4
- discriminator lr from 1e-2 to 1e-3
- entropy mult from 0.01 to 0.0
- target label from 1.0 to 0.9

#8
- entropy mult from 0.0 to 0.01
- rope theta from 500 to 200
- discriminator
    - hidden size from 16 to 256
    - intermeidate size from 64 to 1024
    - layers from 1 to 8
    - attention heads from 4 to 16

#9
- generator steps 8 times per discriminator step
- back to cross-entropy loss

#10
- back to MSE loss (doesn't make a difference since discriminator is frozen)
- target label from 0.9 to 1.0
- generator learning rate from 1e-4 to 1e-3
- gamma from 0.5 to 0.99
    - thought this might give shorter generations more weight
    - can anneal this?
- ran without updating the discriminator to check if the generator can learn to exploit it
    - reward is currently slowly going up. ~0.1 now
    - ~0.25 after 18 epochs (16 steps per epoch) 
    - plateaus ~0.28-0.29 for a while
    - consistently hitting 0.30 after 32 epochs
    - ~0.36 after 100 epochs
    - stopped
    - outputs look like mode collapse?

#11
- remove learning rate scheduler
- momentum back to adamw default (0.9)
- generator learning rate from 1e-3 to 3e-3
- entropy mult from 0.01 to 0.0
- reward still gets stuck ~0.35

#12
- gamma from 0.99 to 0.9
- generator learning rate from 3e-3 to 1e-3
- reward makes it to 0.4 after 2 epochs
- plateaued around 0.45 after 15 epochs

#13
- gamma from 0.9 to 0.2
- reward plateaued around 0.21

#14
- gamma from 0.2 to 0.5
- reward plateaued around 0.36

#15
- gamma from 0.5 to 0.7
- reward plateaued around 0.24
- maybe longer context length requires lower gamma? scaling curve!

#16
- gamma from 0.7 to 0.9
- momentum from 0.9 to 0.5
- adamw beta2 from 0.999 to 0.95

#17
- generator learning rate from 1e-3 to 3e-4
- momentum from 0.5 to 0.9
- reward plat around 0.35

#18
- adamw beta2 from 0.95 to 0.999
- a lot of this is probably getting a lucky seed
- reward plat around 0.3. maybe lower beta2 is better

#19
- generator learning rate from 3e-4 to 1e-3
- adamw beta2 from 0.999 to 0.95
- reward plat in mid 30s

#20
- discriminator learning again
- true data with only one sentence
- batch size from 256 to 128
- don't think it changed but discriminator learning rate is 1e-3
- average reward is 0.8 after one discriminator step???
- discriminator loss exploded to 5 on second step. reward to -0.8
- random large spikes in reward when it gets on a good generation
- then reward plummets when the discriminator learns from this
- reward plat around 0 (went a lot higher) after 347 epochs

#21
- add attention mask to discriminator
- change to left padding for discriminator
- reward jumped to 1 with a random string of mostly 'm'

#22
- generator lr to 1e-4
- discriminator lr to 1e-2
- loss function to binary_cross_entropy_with_logits
- generator is learning too quickly! (maybe)
- losses and reward are oscillating like crazy
    - adding learning rate scheduling again would probably help
    - or a lower gradient norm clipping value
    - maybe that's a good thing though?
- 
