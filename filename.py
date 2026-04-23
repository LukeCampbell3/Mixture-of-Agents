# Write_file filename.py
def doubly_linked_list():
    """
    Define a class for a doubly linked list.
    
    Attributes:
    - head: A reference to the first node in the list (None if the list is empty).
    - tail: A reference to the last node in the list (None if the list is empty).
    """
    def __init__(self):
        self.head = None
        self.tail = None

    def insert_at_beginning(self, value):
        new_node = Node(value)
        
        # If the list is empty, make the new node both head and tail.
        if self.head is None:
            self.head = new_node
            self.tail = new_node
        
        # Otherwise, set the new node's next to current head and update head.
        else:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node

# Show the code in a markdown block
