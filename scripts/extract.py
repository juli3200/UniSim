import struct

class World:
    def __init__(self, file):
        self.end = False

        with open(file, 'rb') as f:
            self.bytes = f.read()
        print(len(self.bytes))
        self.header_size = int(self.bytes[0])
        self.save_jumper = struct.unpack('I', self.bytes[1:5])[0]
        self.time = struct.unpack('Q', self.bytes[5:13])[0]

        self.dimx = struct.unpack('I', self.bytes[13:17])[0]
        self.dimy = struct.unpack('I', self.bytes[17:21])[0]
        self.spawn_size = struct.unpack('f', self.bytes[21:25])[0]
        self.store_capacity = struct.unpack('I', self.bytes[25:29])[0]
        self.fps = struct.unpack('f', self.bytes[29:33])[0]
        self.velocity = struct.unpack('f', self.bytes[33:37])[0]
        self.gravity = [struct.unpack('f', self.bytes[37:41])[0], struct.unpack('f', self.bytes[41:45])[0]]
        self.drag = struct.unpack('f', self.bytes[45:49])[0]

        # reserved bytes 32 bytes from 49 to 81

        self.ligands_per_entity = struct.unpack('I', self.bytes[81:85])[0]
        self.receptors_per_entity = struct.unpack('I', self.bytes[85:89])[0]

        # reserved bytes 32 bytes from 89 to 121

        self.entity_bytes_0 = int(self.bytes[121])
        self.entity_bytes_1 = int(self.bytes[122])
        self.ligand_bytes_0 = int(self.bytes[123])
        self.ligand_bytes_1 = int(self.bytes[124])

        self.protein_n = int(self.bytes[125])


        self.counter = 0

    def get_state(self, n= None):
        if n is None:
            if self.end:
                return None
        
            state = State(self, self.counter)
            self.counter += 1
            return state

        else:
            return State(self, n)


class State:

    def __init__(self, world: World, n: int):
        self.entities = []
        self.ligands = []
        self.world = world
        self.n = n

        try: 
            jumper_address = self.world.header_size + self.n * 4
            address = struct.unpack('I', self.world.bytes[jumper_address:jumper_address + 4])[0]


            self.entities = []
            self.ligands = []


            size = struct.unpack('I', self.world.bytes[address:address + 4])[0]


            info = int(self.world.bytes[address + 4])
            save = info & 0b10  # Extract the first bit of info
            self.genome_save = bool(info & 0b01)  # Extract the second bit of info
        
            time = struct.unpack('f', self.world.bytes[address + 5:address + 9])[0]

            entity_n = struct.unpack('I', self.world.bytes[address + 9:address + 13])[0]
            index = address + 13
            for _ in range(entity_n):
                e = Entity(self.world.bytes, index, world.protein_n, self)
                index+= e.bytes_size
                self.entities.append(e)

            ligand_n = struct.unpack('I', self.world.bytes[index:index + 4])[0]
            index += 4

            for _ in range(ligand_n):
                self.ligands.append(Ligand(self.world.bytes, index))
                index += world.ligand_bytes_0

            
                

        except (struct.error, IndexError):
            self.world.end = True
            return None
        


class Entity:
    def __init__(self, bytes, index, protein_n, parent):
        old_index = index

        self.x = struct.unpack('f', bytes[index:index + 4])[0]
        self.y = struct.unpack('f', bytes[index + 4:index + 8])[0]
        self.velx = struct.unpack('f', bytes[index + 8:index + 12])[0]
        self.vely = struct.unpack('f', bytes[index + 12:index + 16])[0]
        self.size = struct.unpack('f', bytes[index + 16:index + 20])[0]
        self.energy = struct.unpack('f', bytes[index + 20:index + 24])[0]

        self.id = struct.unpack('I', bytes[index + 24:index + 28])[0]

        self.inner_protein_levels = []
        for i in range(protein_n):
            level = struct.unpack('h', bytes[index + 28 + i * 2:index + 30 + i * 2])[0]
            self.inner_protein_levels.append(level)

        index += 28 + protein_n * 2 

        
        # getting received ligands angles in degrees (0 - 180)
        received_n = struct.unpack('I', bytes[index:index + 4])[0]

        index += 4
        self.received_ligands = [int(bytes[index + j]) for j in range(received_n)]

        index += received_n    


        if parent.genome_save:
            self.genome = Genome(bytes, index, parent.world.receptors_per_entity, parent.world.ligands_per_entity)
            index += self.genome.size
        

        self.bytes_size = index - old_index

    def get_position(self):
        return [self.x, self.y]

class Genome:
    def __init__(self, bytes, index, receptors_n, ligand_n):
        old = index

        self.move_threshold = struct.unpack('h', bytes[index:index + 2])[0]
        index += 2
        self.ligands_threshold = struct.unpack('h', bytes[index:index + 2])[0]
        index += 2


        self.ligands = []
        for j in range(ligand_n):
            ligand = struct.unpack('h', bytes[index + j * 2:index + j * 2 + 2])[0]
            self.ligands.append(ligand)
        index += ligand_n * 2

        self.receptors = []
        for j in range(receptors_n):
            receptor = struct.unpack('Q', bytes[index + j * 8:index + j * 8 + 8])[0]
            self.receptors.append(receptor)

        index += receptors_n * 8


        self.size = 4 + (receptors_n * 8) + (ligand_n * 2)
        


class Ligand:
    def __init__(self, bytes, index):
        self.x = struct.unpack('f', bytes[index:index + 4])[0]
        self.y = struct.unpack('f', bytes[index + 4:index + 8])[0]
        
    
    def get_position(self):
        return [self.x, self.y]

