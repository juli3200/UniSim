import struct

class World:
    def __init__(self, file):
        with open(file, 'rb') as f:
            self.bytes = f.read()
        print(len(self.bytes))
        self.header_size = int(self.bytes[0])
        self.save_jumper = struct.unpack('I', self.bytes[1:5])[0]
        self.time = struct.unpack('Q', self.bytes[5:13])[0]

        self.dimy = struct.unpack('I', self.bytes[13:17])[0]
        self.dimx = struct.unpack('I', self.bytes[17:21])[0]
        self.spawn_size = struct.unpack('f', self.bytes[21:25])[0]
        self.store_capacity = struct.unpack('I', self.bytes[25:29])[0]
        self.fps = struct.unpack('f', self.bytes[29:33])[0]
        self.velocity = struct.unpack('f', self.bytes[33:37])[0]
        self.gravity = [struct.unpack('f', self.bytes[37:41])[0], struct.unpack('f', self.bytes[41:45])[0]]
        self.friction = struct.unpack('f', self.bytes[45:49])[0]

        self.entity_bytes_0 = int(self.bytes[49])
        self.entity_bytes_1 = int(self.bytes[50])
        self.ligand_bytes_0 = int(self.bytes[51])
        self.ligand_bytes_1 = int(self.bytes[52])

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



            #size = struct.unpack('I', self.world.bytes[address:address + 4])[0]
            #print("size:", size)
            size = struct.unpack('I', self.world.bytes[address:address + 4])[0]
            save = int(self.world.bytes[address + 4])



            time = struct.unpack('f', self.world.bytes[address + 5:address + 9])[0]


            entity_n = struct.unpack('I', self.world.bytes[address + 9:address + 13])[0]

            for i in range(entity_n):
                index = address + 13 + i * self.world.entity_bytes_0
                self.entities.append(Entity(self.world.bytes, index))

            ligand_n = struct.unpack('I', self.world.bytes[address + 13 + entity_n * self.world.entity_bytes_0:address + 17 + entity_n * self.world.entity_bytes_0])[0]

            for i in range(ligand_n):
                index = address + 17 + entity_n * self.world.entity_bytes_0 + i * self.world.ligand_bytes_0
                self.ligands.append(Ligand(self.world.bytes, index))
            
                

        except (struct.error, IndexError):
            print("No more saves")
            return None
        


class Entity:
    def __init__(self, bytes, index):
        self.x = struct.unpack('f', bytes[index:index + 4])[0]
        self.y = struct.unpack('f', bytes[index + 4:index + 8])[0]
        self.size = struct.unpack('f', bytes[index + 8:index + 12])[0]
        self.velx = struct.unpack('f', bytes[index + 12:index + 16])[0]
        self.vely = struct.unpack('f', bytes[index + 16:index + 20])[0]



    def get_position(self):
        return [self.x, self.y]


class Ligand:
    def __init__(self, bytes, index):
        self.x = struct.unpack('f', bytes[index:index + 4])[0]
        self.y = struct.unpack('f', bytes[index + 4:index + 8])[0]
        self.message = int(bytes[index + 12])
    
    def get_position(self):
        return [self.x, self.y]

