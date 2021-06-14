package queue_singlelinkedlist;
import java.util.*;

public class FifoQueue<E> extends AbstractQueue<E> implements Queue<E> {
	private QueueNode<E> last;
	private int size;

	public FifoQueue() {
		super();
		last = null;
		size = 0;
	}

	/**	
	 * Inserts the specified element into this queue, if possible
	 * post:	The specified element is added to the rear of this queue
	 * @param	e the element to insert
	 * @return	true if it was possible to add the element 
	 * 			to this queue, else false
	 */
	public boolean offer(E e) {
		QueueNode<E> newLastNode = new QueueNode<E>(e);
		
		if (size == 0) {
			last = newLastNode;
			last.next = last; 
			size++;
			return true;
		}
		if (size == 1) {
			last.next = newLastNode; 
			newLastNode.next = last;
			last = newLastNode;
			size++;
			return true;
		}
		
		QueueNode<E> firstNode = last.next;  
		last.next = newLastNode;
		last = newLastNode;
		newLastNode.next = firstNode; 
		size++;
		
		return true;
	}
	
	/**	
	 * Returns the number of elements in this queue
	 * @return the number of elements in this queue
	 */
	public int size() {		
		return size;
	}
	
	/**	
	 * Retrieves, but does not remove, the head of this queue, 
	 * returning null if this queue is empty
	 * @return 	the head element of this queue, or null 
	 * 			if this queue is empty
	 */
	public E peek() {
		if (size == 0) {
			return null;
		}
		
		return last.next.element;
	}

	/**	
	 * Retrieves and removes the head of this queue, 
	 * or null if this queue is empty.
	 * post:	the head of the queue is removed if it was not empty
	 * @return 	the head of this queue, or null if the queue is empty 
	 */
	public E poll() {
		if (size == 0) {
			return null;
		}
		if (size == 1) {
			QueueNode<E> head = last;
			last = null;
			size--;
			return head.element;
		}
		
		QueueNode<E> head = last.next;
		last.next = head.next; 
		size--;
		return head.element;
	}
	
	/**	
	 * Returns an iterator over the elements in this queue
	 * @return an iterator over the elements in this queue
	 */	
	public Iterator<E> iterator() {
		return new QueueIterator();
	}
	
	/**
     * Appends the specified queue to this queue
     * post: all elements from the specified queue are appended
     * to this queue. The specified queue (q) is empty after the call.
     * @param q the queue to append
     * @throws IllegalArgumentException if this queue and q are identical
     */
	public void append(FifoQueue<E> q) {
		// om samma icke-tomma lista
		if (this.equals(q)) {
			throw new IllegalArgumentException();
		}
		
		// om någon av listorna är tomma
		if (size == 0) {
			last = q.last;
			size = q.size; 
			
			q.last = null;
			q.size = 0;
			return; 
		}
		if (q.size == 0) {
			return; 
		}
		
		QueueNode<E> head = last.next;
		QueueNode<E> qHead = q.last.next; 
		last.next = qHead;
		last = q.last;
		last.next = head; 
		size += q.size; 
		
		q.last = null;
		q.size = 0 ; 
	}
	
	private static class QueueNode<E> {
		E element;
		QueueNode<E> next;

		private QueueNode(E x) {
			element = x;
			next = null;
		}
	}
	
	private class QueueIterator implements Iterator<E> {
		private QueueNode<E> currentNode;
		private int counter; // håll reda på antal gånger next() anropats, eftersom listan är cirkulär
		
		private QueueIterator() {
			counter = 0;
			
			if (size == 0) {
				currentNode = null;
			} else {
				currentNode = last.next; 
			}
		}
		
		@Override
		public boolean hasNext() {
			// tom lista
			if (size == 0) {
				return false; 
			}
			// om listan har gåtts igenom
			if (counter == size) {
				return false; 
			}
			
			return true;
		}

		@Override
		public E next() {
			if (!hasNext()) {
				throw new NoSuchElementException();
			}
			
			QueueNode<E> temp = currentNode;
			currentNode = currentNode.next;
			counter++;
			return temp.element;
		}
		
	}

}
