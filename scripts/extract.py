import struct

class World:
    def __init__(self, file):
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
        self.friction = struct.unpack('f', self.bytes[45:49])[0]

        self.ligands_per_entity = struct.unpack('I', self.bytes[49:53])[0]
        self.receptors_per_entity = struct.unpack('I', self.bytes[53:57])[0]

        self.entity_bytes_0 = int(self.bytes[57])
        self.entity_bytes_1 = int(self.bytes[58])
        self.ligand_bytes_0 = int(self.bytes[59])
        self.ligand_bytes_1 = int(self.bytes[60])

        self.protein_n = int(self.bytes[61])

        self.counter = 0

    def get_state(self, n= None):
        if n is None:
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

            info = int(self.world.bytes[address])
            save = info & 0b01  # Extract the first bit of info
            self.genome_save = bool((info >> 1) & 0b01)  # Extract the second bit of info

            size = struct.unpack('I', self.world.bytes[address+1:address + 5])[0]

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
            return None
        


class Entity:
    def __init__(self, bytes, index, protein_n, parent):
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

        i = 28 + protein_n * 2
        received_n = struct.unpack('I', bytes[index + i:index + i + 4])[0]

        # getting received ligands angles in degrees (0 - 180)
        i += 4
        self.received_ligands = [int(bytes[index + j]) for j in range(i, i + received_n)]

        if parent.genome_save:
            i += received_n
            self.genome = Genome(bytes, index + i, parent.world.receptors_per_entity, parent.world.ligands_per_entity)

        self.bytes_size = i + (self.genome.size if parent.genome_save else 0)

    def get_position(self):
        return [self.x, self.y]

class Genome:
    def __init__(self, bytes, index, receptors_n, ligand_n):

        self.move_threshold = struct.unpack('h', bytes[index:index + 2])[0]
        index += 2
        self.ligands_threshold = struct.unpack('h', bytes[index:index + 2])[0]
        index += 2

        self.receptors = []
        for i in range(receptors_n):
            receptor = struct.unpack('h', bytes[index + i * 2:index + i * 2 + 2])[0]
            self.receptors.append(receptor)

        index += receptors_n * 2
        self.ligands = []
        for i in range(ligand_n):
            ligand = struct.unpack('h', bytes[index + i * 2:index + i * 2 + 2])[0]
            self.ligands.append(ligand)
        
        self.size = 4 + (receptors_n + ligand_n) * 2
        


class Ligand:
    def __init__(self, bytes, index):
        self.x = struct.unpack('f', bytes[index:index + 4])[0]
        self.y = struct.unpack('f', bytes[index + 4:index + 8])[0]
        
    
    def get_position(self):
        return [self.x, self.y]

