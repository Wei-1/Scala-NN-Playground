import akka.actor.ActorSystem
import org.scalatestplus.play._
import play.api.test.Helpers._
import play.api.test.FakeRequest

import controllers.PgController
import com.interplanetarytech.nn._

/**
 * Unit tests can run without a full Play application.
 */
class UnitSpec extends PlaySpec {

  "PgController" should {

    "return a valid error with action init" in {
      val pg = new Playground
      val controller = new PgController(stubControllerComponents(), pg)
      val result = controller.init(FakeRequest())
      (contentAsString(result).toDouble < 1) mustBe true
      (contentAsString(result).toDouble > 0) mustBe true
    }
  }

  "Playground" should {
  }

}
