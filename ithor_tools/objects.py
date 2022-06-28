object_bed = [
    'AlarmClock','Book', "CellPhone",'CreditCard',"KeyChain", "Pillow","CD","Laptop",
    "TeddyBear","TissueBox"]
    
object_kitchen = ['Book', 'Apple', 'Bread', "DishSponge" ,"Kettle","Pot", "Bowl", "Pan","Toaster","PaperTowelRoll"]

object_living_room = ['Book',"CellPhone",'CreditCard',"KeyChain", "RemoteControl","TissueBox","Laptop"]

object_bath = ['HandTowel',"SoapBar","SprayBottle","TissueBox","ToiletPaper","Towel"]

total = list(set(object_bed+object_kitchen+object_living_room+object_bath))

# total = ['AlarmClock', 'Apple', 'BaseballBat', 'BasketBall', 'Bowl', 'GarbageCan', 'HousePlant', 
# 'Laptop', 'Mug', 'SprayBottle', 'Television', 'Vase']

def get_obj_list(scene_type):
    if scene_type == 'all':
        return total
    elif scene_type == 'bed':
        return object_bed
    elif scene_type == 'kitchen':
        return object_kitchen
    elif scene_type == 'living_room':
        return object_living_room
    elif scene_type == 'bath':
        return object_bath
    else:
        raise NotImplementedError

def choose_query_objects(objects,scene_type='all'):
    object_list= get_obj_list(scene_type)
    query_objects = []
    for obj in objects:
        if obj['objectType'] in object_list:
            query_objects.append(obj)
    return query_objects