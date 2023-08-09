# Phantoms deformations 

## Shepp-Loggan :
Let $c$ be the center of the image. The deformations, labeled as 1 and 2, can be described using an affine model: $$u_t(x) = c + \left(s(t) \hspace{3mm} 0 \atop 0 \hspace{3mm} s(t)^{-1} \right) (x - c)$$ 
	
- deformation 1: This deformation, defined by $s(t) = 0.05 \cos(\frac{2 \pi t}{T}) + 0.95$ is used to represent a periodic motion.

- deformation 2: The second deformation $s(t)$ is modeled to be affine by parts. This is used to represent the piston cycle applied to deform the lamb brain as shown in the image below: 
![cycle](https://github.com/Tommte/sp_moco_demo/blob/main/animations/zfigs/carotid_amplitude.png)

	
## White spot :
The white spot is displaced along both diagonals of the image. The displacement vector used is (18, 18) pixels.
