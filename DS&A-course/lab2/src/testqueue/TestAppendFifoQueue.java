package testqueue;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.jupiter.api.Assertions.*;

import java.util.Iterator;
import java.util.Queue;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import queue_singlelinkedlist.FifoQueue;

class TestAppendFifoQueue {
	private FifoQueue<Integer> myIntQueue;
	private FifoQueue<Integer> anotherIntQueue;
	
	@BeforeEach
	void setUp() {
		myIntQueue = new FifoQueue<Integer>();
		anotherIntQueue = new FifoQueue<Integer>();
	}

	@AfterEach
	void tearDown() {
		myIntQueue = null;
	}
	
	@Test
	void testTwoEmptyQueues() {
		myIntQueue.append(anotherIntQueue); 
		assertEquals(0, myIntQueue.size(), "Size nonzero after appended two zero length queues"); 	
		assertEquals(0, anotherIntQueue.size(), "Wrong size of appended queue");
	}
	
	@Test
	void testEmptyQueueAppendedToNonemptyQueue() {
		FifoQueue<Integer> emptyQueue = new FifoQueue<Integer>();
		
		for (int i = 0; i < 5; i++) {
			myIntQueue.offer(i);
			anotherIntQueue.offer(i);
		}
		
		myIntQueue.append(emptyQueue);
		assertEquals(0, emptyQueue.size(), "Wrong size of appended queue");
		
		// kontrollera att listan förblir oförändrad
		assertEquals(anotherIntQueue.size(), myIntQueue.size(), "Wrong size after append of empty queue");
		
		Iterator<Integer> myIter = myIntQueue.iterator();
		Iterator<Integer> anotherIter = anotherIntQueue.iterator();
		
		for (int i = 0; i < myIntQueue.size(); i++) {
			assertTrue(myIter.hasNext(), "Iterator does not have next");
			assertEquals(myIter.next(), anotherIter.next(), "Empty queue appended changed the queue on which it was appended");
		}
		assertFalse(myIter.hasNext(), "Non-empty iterator after full iteration");
	}
	
	@Test
	void testNonemptyQueueAppendedToEmptyQueue() {
		FifoQueue<Integer> emptyQueue = new FifoQueue<Integer>();
		
		for (int i = 0; i < 5; i++) {
			myIntQueue.offer(i);
			anotherIntQueue.offer(i);
		}
		
		emptyQueue.append(myIntQueue); 
		assertEquals(0, myIntQueue.size(), "Wrong size of appended queue");
		
		// kontrollera att listan förblir oförändrad
		assertEquals(anotherIntQueue.size(), emptyQueue.size(), "Wrong size after append of empty queue");
		
		Iterator<Integer> myIter = emptyQueue.iterator();
		Iterator<Integer> anotherIter = anotherIntQueue.iterator();
		
		for (int i = 0; i < emptyQueue.size(); i++) {
			assertTrue(myIter.hasNext(), "Iterator does not have another element");	
			assertEquals(myIter.next(), anotherIter.next(), "Empty queue appended changed the queue on which it was appended");
		}
		assertFalse(myIter.hasNext(), "Non-empty iterator after full iteration");
	}
	
	@Test
	void testTwoNonemptyQueues() {
		FifoQueue<Integer> correctQueue = new FifoQueue<Integer>();

		for (int i = 0; i < 5; i++) {
			anotherIntQueue.offer(i);
			correctQueue.offer(i);
		}
		myIntQueue.append(anotherIntQueue);
		assertEquals(0, anotherIntQueue.size(), "Wrong size of appended queue");
		
		// kontrollera att de är samma (elementvis)
		assertEquals(myIntQueue.size(), correctQueue.size(), "Wrong size after append");
		
		Iterator<Integer> iter = myIntQueue.iterator(); 
		for (int i = 0; i < myIntQueue.size(); i++) {
			assertTrue(iter.hasNext(), "Iterator fault");
			assertEquals(i, iter.next(), "Not correctly appended");
		}
		assertFalse(iter.hasNext(), "iterator not empty after complete iteration");
	}
	
	@Test
	void testAppendSelf() {
		assertThrows(IllegalArgumentException.class, () -> myIntQueue.append(myIntQueue)); 
	}

}
