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
- should bump batch size back up again

#23
- disable discriminator learning
- generator lr to 1e-3
- reward plateau ~0.15

#24
- switch to MSE loss
- reward oscillates a lot more
- generation loss depends a lot on the first token
    - especially when exploiting a randomly initialized discriminator
    - bigger batch size should mitigate this
- learning doesn't plateau as hard as before
- reward curve is shaped like a growing sinusoid
    - this might be why lower momentum is better
    - remember that RL in non-stationary
- reward plat ~0.39

#25
- batch size from 128 to 256
- switch to BCE loss
- gamma from 0.9 to 0.25
- matches previous after 1.5 epochs
- only plateaus a bit higher than previous

#26
- generator lr to 1e-4
- plateau around 0.25

#27
- gamma from 0.25 to 0.9
- stopped around 0.36. starting to plateau

#28
- enable discriminator learning
- converges towards only padding tokens

#29
- entropy mult to 0.01
- discriminator accumulation to 2
- top-p sampling to 0.95
- remove option of generating token 0 in generator
- discriminator loss is smoother now
- average reward around -0.5, but learning feels better
- still gibberish generations

#30
- discriminator momentum to 0.5
- discriminator beta2 to 0.95
- discriminator accumulation to 8
- loss and reward back to oscillating

#31
- discriminator accumulation to 4
- LR scheduler back on
- discriminator momentum cycle 0.85 -> 0.95
- generator momentum cycle 0.2 -> 0.6
- generator learning very slow
    - maybe increase its LR
    - or reduce discriminator LR
- after 74 epochs: discriminator seems to have fully learned. generator did not

#32
- generator lr to 3e-4
- discriminator lr to 3e-3
- discriminator accumulation to 8
- max grad norm to 0.5
- entropy mult to 0.0
- learning rate scheduling off
- try this again

#33
- simpler data
- this seems to be working well

#34
- even simpler data
- context length to 4
- hit one good generation and reward went from -0.7 to -0.1 then leveled out again at -0.5
    - this is why low momentum
- generator is learning and outputs are decently close to the training data
    - letters are correct but sometimes in weird orders
        - lower gamma may help fix this
        - or lower rope theta
        - or absolute embeddings
- plateauing around -0.47 reward. will keep training running though
- discriminator is overpowering the generator, but only slightly

#35
- rope theta from 200 to 25
- discriminator attention dropout from 0.5 to 0.0
- generations frequently repeat letters
- similar plateau to previous

#36
- context length to 8
- attention heads in both networks to 4
- rope theta to 50
- similar

#37
- slightly more complex data
- generator layers from 3 to 8
- discriminator
    - layers from 6 to 4
    - hidden size from 256 to 64
    - intermediate size from 1024 to 256
- generator does worse now
    - might be because of the data
    - more diverse data would probably make the discriminator learn slower
        - scaling?
- reward plat -0.83, disc loss 0.26

#38
- top-p from 0.95 to 1.0
- entropy mult from 0.0 to 0.1
- reward plateau -0.5, disc loss 0.47

#39
- divide discriminator loss by accumulation steps
- divide both losses by context length

#40
- remove division by context length
- batch size got raised to 1024 a few experiments ago. forgot to record when

#41
- batch size to 128
- context length to 32
- new data (karpathy names.txt)
- allow generator to choose EOS token again
- attention heads to 8 in both models
- generator 8 layers to 6 layers
- discriminator 4 layers to 3 layers
- it's learning pretty well. not fully converged after ~50 epochs

#42
- remove division by accumulation steps in discriminator loss
- discriminator lr from 3e-3 to 1.5e-3
- generator lr from 3e-4 to 6e-4
- discriminator momentum from 0.85 to 0.9
- generator momentum from 0.2 to 0.5
- discriminator learning a bit too slow?
- is higher entropy stabilizing the learning? (0.1)

#43
- entropy mult from 0.1 to 0.01
- generator lr from 6e-4 to 3e-4

#44
- entropy mult from 0.01 to 0.1
- after 112 epochs: reward -0.15, disc loss 0.61

#45
- gamma from 0.9 to 0.5
- rewards are hovering around a lower value in early training (-0.4 instead of -0.1)
- less repetition of EOS token. could be luck?
- generator is learning slower now since returns are lower
- 

#46
- rope theta from 50 to 200
- re-add shift to data